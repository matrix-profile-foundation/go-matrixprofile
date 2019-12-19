// Package matrixprofile computes the matrix profile and matrix profile index of a time series
package matrixprofile

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
	"strconv"
	"sync"

	"github.com/matrix-profile-foundation/go-matrixprofile/av"
	"github.com/matrix-profile-foundation/go-matrixprofile/method"
	"github.com/matrix-profile-foundation/go-matrixprofile/util"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/fourier"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
	"gonum.org/v1/plot/vg/vgimg"
)

// MatrixProfile is a struct that tracks the current matrix profile computation
// for a given timeseries of length N and subsequence length of M. The profile
// and the profile index are stored here.
type MatrixProfile struct {
	A        []float64    // query time series
	B        []float64    // timeseries to perform full join with
	AMean    []float64    // sliding mean of a with a window of m each
	AStd     []float64    // sliding standard deviation of a with a window of m each
	BMean    []float64    // sliding mean of b with a window of m each
	BStd     []float64    // sliding standard deviation of b with a window of m each
	BF       []complex128 // holds an existing calculation of the FFT of b timeseries
	N        int          // length of the timeseries
	M        int          // length of a subsequence
	SelfJoin bool         // indicates whether a self join is performed with an exclusion zone
	MP       []float64    // matrix profile
	Idx      []int        // matrix profile index
	AV       av.AV        // type of annotation vector which defaults to all ones
}

// New creates a matrix profile struct with a given timeseries length n and
// subsequence length of m. The first slice, a, is used as the initial
// timeseries to join with the second, b. If b is nil, then the matrix profile
// assumes a self join on the first timeseries.
func New(a, b []float64, m int) (*MatrixProfile, error) {
	if a == nil || len(a) == 0 {
		return nil, fmt.Errorf("first slice is nil or has a length of 0")
	}

	if b != nil && len(b) == 0 {
		return nil, fmt.Errorf("second slice must be nil for self-join operation or have a length greater than 0")
	}

	mp := MatrixProfile{
		A: a,
		M: m,
		N: len(b),
	}
	if b == nil {
		mp.N = len(a)
		mp.B = a
		mp.SelfJoin = true
	} else {
		mp.B = b
	}

	if mp.M*2 >= mp.N {
		return nil, fmt.Errorf("subsequence length must be less than half the timeseries")
	}

	if mp.M < 2 {
		return nil, fmt.Errorf("subsequence length must be at least 2")
	}

	if err := mp.initCaches(); err != nil {
		return nil, err
	}

	mp.MP = make([]float64, mp.N-mp.M+1)
	mp.Idx = make([]int, mp.N-m+1)
	for i := 0; i < len(mp.MP); i++ {
		mp.MP[i] = math.Inf(1)
		mp.Idx[i] = math.MaxInt64
	}

	mp.AV = av.Default

	return &mp, nil
}

// initCaches initializes cached data including the timeseries a and b rolling mean
// and standard deviation and full fourier transform of timeseries b
func (mp *MatrixProfile) initCaches() error {
	var err error
	// precompute the mean and standard deviation for each window of size m for all
	// sliding windows across the b timeseries
	mp.BMean, mp.BStd, err = util.MovMeanStd(mp.B, mp.M)
	if err != nil {
		return err
	}

	mp.AMean, mp.AStd, err = util.MovMeanStd(mp.A, mp.M)
	if err != nil {
		return err
	}

	// precompute the fourier transform of the b timeseries since it will
	// be used multiple times while computing the matrix profile
	fft := fourier.NewFFT(mp.N)
	mp.BF = fft.Coefficients(nil, mp.B)

	return nil
}

// crossCorrelate computes the sliding dot product between two slices
// given a query and time series. Uses fast fourier transforms to compute
// the necessary values. Returns the a slice of floats for the cross-correlation
// of the signal q and the mp.B signal. This makes an optimization where the query
// length must be less than half the length of the timeseries, b.
func (mp MatrixProfile) crossCorrelate(q []float64, fft *fourier.FFT) []float64 {
	qpad := make([]float64, mp.N)
	for i := 0; i < len(q); i++ {
		qpad[i] = q[mp.M-i-1]
	}
	qf := fft.Coefficients(nil, qpad)

	// in place multiply the fourier transform of the b time series with
	// the subsequence fourier transform and store in the subsequence fft slice
	for i := 0; i < len(qf); i++ {
		qf[i] = mp.BF[i] * qf[i]
	}

	dot := fft.Sequence(nil, qf)

	for i := 0; i < mp.N-mp.M+1; i++ {
		dot[mp.M-1+i] = dot[mp.M-1+i] / float64(mp.N)
	}
	return dot[mp.M-1:]
}

// mass calculates the Mueen's algorithm for similarity search (MASS)
// between a specified query and timeseries. Writes the euclidean distance
// of the query to every subsequence in mp.B to profile.
func (mp MatrixProfile) mass(q []float64, profile []float64, fft *fourier.FFT) error {
	qnorm, err := util.ZNormalize(q)
	if err != nil {
		return err
	}

	dot := mp.crossCorrelate(qnorm, fft)

	// converting cross correlation value to euclidian distance
	for i := 0; i < len(dot); i++ {
		profile[i] = math.Sqrt(math.Abs(2 * (float64(mp.M) - (dot[i] / mp.BStd[i]))))
	}
	return nil
}

// distanceProfile computes the distance profile between a and b time series.
// If b is set to nil then it assumes a self join and will create an exclusion
// area for trivial nearest neighbors. Writes the euclidean distance between
// the specified subsequence in mp.A with each subsequence in mp.B to profile
func (mp MatrixProfile) distanceProfile(idx int, profile []float64, fft *fourier.FFT) error {
	if idx > len(mp.A)-mp.M {
		return fmt.Errorf("provided index  %d is beyond the length of timeseries %d minus the subsequence length %d", idx, len(mp.A), mp.M)
	}

	if err := mp.mass(mp.A[idx:idx+mp.M], profile, fft); err != nil {
		return err
	}

	// sets the distance in the exclusion zone to +Inf
	if mp.SelfJoin {
		util.ApplyExclusionZone(profile, idx, mp.M/2)
	}
	return nil
}

// calculateDistanceProfile converts a sliding dot product slice of floats into
// distances and normalizes the output. Writes results back into the profile slice
// of floats representing the distance profile.
func (mp MatrixProfile) calculateDistanceProfile(dot []float64, idx int, profile []float64) error {
	if idx > len(mp.A)-mp.M {
		return fmt.Errorf("provided index %d is beyond the length of timeseries a %d minus the subsequence length %d", idx, len(mp.A), mp.M)
	}

	if len(profile) != len(dot) {
		return fmt.Errorf("profile length, %d, is not the same as the dot product length, %d", len(profile), len(dot))
	}

	// converting cross correlation value to euclidian distance
	for i := 0; i < len(dot); i++ {
		profile[i] = math.Sqrt(2 * float64(mp.M) * math.Abs(1-(dot[i]-float64(mp.M)*mp.BMean[i]*mp.AMean[idx])/(float64(mp.M)*mp.BStd[i]*mp.AStd[idx])))
	}

	if mp.SelfJoin {
		// sets the distance in the exclusion zone to +Inf
		util.ApplyExclusionZone(profile, idx, mp.M/2)
	}
	return nil
}

// stmp computes the full matrix profile given two time series as inputs.
// If the second time series is set to nil then a self join on the first
// will be performed. Stores the matrix profile and matrix profile index
// in the struct.
func (mp *MatrixProfile) stmp() error {
	var err error
	profile := make([]float64, mp.N-mp.M+1)

	fft := fourier.NewFFT(mp.N)
	for i := 0; i < mp.N-mp.M+1; i++ {
		if err = mp.distanceProfile(i, profile, fft); err != nil {
			return err
		}

		for j := 0; j < len(profile); j++ {
			if profile[j] <= mp.MP[j] {
				mp.MP[j] = profile[j]
				mp.Idx[j] = i
			}
		}
	}

	return nil
}

// stamp uses random ordering to compute the matrix profile. User can specify the
// sample to be anything between 0 and 1 so that the computation early terminates
// and provides the current computed matrix profile. 1 represents the exact matrix
// profile. This should compute far faster at the cost of an approximation of the
// matrix profile. Stores the matrix profile and matrix profile index in the struct.
func (mp *MatrixProfile) stamp(sample float64, parallelism int) error {
	if sample <= 0.0 {
		return fmt.Errorf("must provide a sampling greater than 0 and at most 1, sample: %.3f", sample)
	}

	randIdx := rand.Perm(len(mp.A) - mp.M + 1)

	batchSize := (len(mp.A)-mp.M+1)/parallelism + 1
	results := make([]chan *mpResult, parallelism)
	for i := 0; i < parallelism; i++ {
		results[i] = make(chan *mpResult)
	}

	// go routine to continually check for results on the slice of channels
	// for each batch kicked off. This merges the results of the batched go
	// routines by picking the lowest value in each batch's matrix profile and
	// updating the matrix profile index.
	var err error
	done := make(chan bool)
	go func() {
		err = mp.mergeMPResults(results)
		done <- true
	}()

	// kick off multiple go routines to process a batch of rows returning back
	// the matrix profile for that batch and any error encountered
	var wg sync.WaitGroup
	wg.Add(parallelism)
	for batch := 0; batch < parallelism; batch++ {
		go func(idx int) {
			results[idx] <- mp.stampBatch(idx, batchSize, sample, randIdx, &wg)
		}(batch)
	}
	wg.Wait()

	// waits for all results to be read and merged before returning success
	<-done

	return err
}

// stampBatch processes a batch set of rows in a matrix profile calculation
func (mp MatrixProfile) stampBatch(idx, batchSize int, sample float64, randIdx []int, wg *sync.WaitGroup) *mpResult {
	defer wg.Done()
	if idx*batchSize+mp.M > len(mp.A) {
		// got an index larger than mp.A so ignore
		return &mpResult{}
	}

	// initialize this batch's matrix profile results
	result := &mpResult{
		MP:  make([]float64, mp.N-mp.M+1),
		Idx: make([]int, mp.N-mp.M+1),
	}
	for i := 0; i < len(mp.MP); i++ {
		result.MP[i] = math.Inf(1)
		result.Idx[i] = math.MaxInt64
	}

	var err error
	profile := make([]float64, len(result.MP))
	fft := fourier.NewFFT(mp.N)
	for i := 0; i < int(float64(batchSize)*sample); i++ {
		if idx*batchSize+i >= len(randIdx) {
			break
		}
		if err = mp.distanceProfile(randIdx[idx*batchSize+i], profile, fft); err != nil {
			return &mpResult{nil, nil, err}
		}
		for j := 0; j < len(profile); j++ {
			if profile[j] <= result.MP[j] {
				result.MP[j] = profile[j]
				result.Idx[j] = randIdx[idx*batchSize+i]
			}
		}
	}
	return result
}

// Update updates a matrix profile and matrix profile index in place providing streaming
// like behavior.
func (mp *MatrixProfile) Update(newValues []float64) error {
	var err error

	var profile []float64
	for _, val := range newValues {
		// add to the a and b time series and increment the time series length
		if mp.SelfJoin {
			mp.A = append(mp.A, val)
			mp.B = mp.A
		} else {
			mp.B = append(mp.B, val)
		}
		mp.N++

		// increase the size of the Matrix Profile and Index
		mp.MP = append(mp.MP, math.Inf(1))
		mp.Idx = append(mp.Idx, math.MaxInt64)

		if err = mp.initCaches(); err != nil {
			return err
		}

		// only compute the last distance profile
		profile = make([]float64, len(mp.MP))
		fft := fourier.NewFFT(mp.N)
		if err = mp.distanceProfile(len(mp.A)-mp.M, profile, fft); err != nil {
			return err
		}

		minVal := math.Inf(1)
		minIdx := math.MaxInt64
		for j := 0; j < len(profile)-1; j++ {
			if profile[j] <= mp.MP[j] {
				mp.MP[j] = profile[j]
				mp.Idx[j] = mp.N - mp.M
			}
			if profile[j] < minVal {
				minVal = profile[j]
				minIdx = j
			}
		}
		mp.MP[mp.N-mp.M] = minVal
		mp.Idx[mp.N-mp.M] = minIdx
	}
	return nil
}

// mpResult is the output struct from a batch processing for STAMP and STOMP. This struct
// can later be merged together in linear time or with a divide and conquer approach
type mpResult struct {
	MP  []float64
	Idx []int
	Err error
}

// stomp is an optimization on the STAMP approach reducing the runtime from O(n^2logn)
// down to O(n^2). This is an ordered approach, since the sliding dot product or cross
// correlation can be easily updated for the next sliding window, if the previous window
// dot product is available. This should also greatly reduce the number of memory
// allocations needed to compute an arbitrary timeseries length.
func (mp *MatrixProfile) stomp(parallelism int) error {
	batchSize := (len(mp.A)-mp.M+1)/parallelism + 1
	results := make([]chan *mpResult, parallelism)
	for i := 0; i < parallelism; i++ {
		results[i] = make(chan *mpResult)
	}

	// go routine to continually check for results on the slice of channels
	// for each batch kicked off. This merges the results of the batched go
	// routines by picking the lowest value in each batch's matrix profile and
	// updating the matrix profile index.
	var err error
	done := make(chan bool)
	go func() {
		err = mp.mergeMPResults(results)
		done <- true
	}()

	// kick off multiple go routines to process a batch of rows returning back
	// the matrix profile for that batch and any error encountered
	var wg sync.WaitGroup
	wg.Add(parallelism)
	for batch := 0; batch < parallelism; batch++ {
		go func(idx int) {
			results[idx] <- mp.stompBatch(idx, batchSize, &wg)
		}(batch)
	}
	wg.Wait()

	// waits for all results to be read and merged before returning success
	<-done

	return err
}

// stompBatch processes a batch set of rows in matrix profile calculation. Each batch
// will compute its first row's dot product and build the subsequent matrix profile and
// matrix profile index using the stomp iterative algorithm. This also uses the very
// first row's dot product to update the very first index of the current row's
// dot product.
func (mp MatrixProfile) stompBatch(idx, batchSize int, wg *sync.WaitGroup) *mpResult {
	defer wg.Done()
	if idx*batchSize+mp.M > len(mp.A) {
		// got an index larger than mp.A so ignore
		return &mpResult{}
	}

	// compute for this batch the first row's sliding dot product
	fft := fourier.NewFFT(mp.N)
	dot := mp.crossCorrelate(mp.A[idx*batchSize:idx*batchSize+mp.M], fft)

	profile := make([]float64, len(dot))
	var err error
	if err = mp.calculateDistanceProfile(dot, idx*batchSize, profile); err != nil {
		return &mpResult{nil, nil, err}
	}

	// initialize this batch's matrix profile results
	result := &mpResult{
		MP:  make([]float64, mp.N-mp.M+1),
		Idx: make([]int, mp.N-mp.M+1),
	}

	copy(result.MP, profile)
	for i := 0; i < len(profile); i++ {
		result.Idx[i] = idx * batchSize
	}

	// iteratively update for this batch each row's matrix profile and matrix
	// profile index
	var nextDotZero float64
	for i := 1; i < batchSize; i++ {
		if idx*batchSize+i-1 >= len(mp.A) || idx*batchSize+i+mp.M-1 >= len(mp.A) {
			// looking for an index beyond the length of mp.A so ignore and move one
			// with the current processed matrix profile
			break
		}
		for j := mp.N - mp.M; j > 0; j-- {
			dot[j] = dot[j-1] - mp.B[j-1]*mp.A[idx*batchSize+i-1] + mp.B[j+mp.M-1]*mp.A[idx*batchSize+i+mp.M-1]
		}

		// recompute the first cross correlation since the algorithm is only valid for
		// points after it. Previous optimization of using a precomputed cache ONLY applies
		// if we're doing a self-join and is invalidated with AB-joins of different time series
		nextDotZero = 0
		for k := 0; k < mp.M; k++ {
			nextDotZero += mp.A[idx*batchSize+i+k] * mp.B[k]
		}
		dot[0] = nextDotZero
		if err = mp.calculateDistanceProfile(dot, idx*batchSize+i, profile); err != nil {
			return &mpResult{nil, nil, err}
		}

		// element wise min update of the matrix profile and matrix profile index
		for j := 0; j < len(profile); j++ {
			if profile[j] <= result.MP[j] {
				result.MP[j] = profile[j]
				result.Idx[j] = idx*batchSize + i
			}
		}
	}
	return result
}

// mpxBatch processes a batch set of rows in matrix profile calculation.
func (mp MatrixProfile) mpxBatch(idx int, mu, sig, df, dg []float64, batchSize int, wg *sync.WaitGroup) *mpResult {
	defer wg.Done()
	if idx*batchSize+mp.M/4 > mp.N-mp.M+1 {
		// got an index larger than max lag so ignore
		return &mpResult{}
	}

	mpr := &mpResult{
		MP:  make([]float64, mp.N-mp.M+1),
		Idx: make([]int, mp.N-mp.M+1),
	}
	for i := 0; i < len(mpr.MP); i++ {
		mpr.MP[i] = -1
	}

	var c, c_cmp float64
	for diag := idx*batchSize + mp.M/4; diag < (idx+1)*batchSize+mp.M/4; diag++ {
		if diag >= mp.N-mp.M+1 {
			break
		}
		c = 0
		for i := 0; i < mp.M; i++ {
			c += (mp.A[diag+i] - mu[diag]) * (mp.A[i] - mu[0])
		}
		for offset := 0; offset < mp.N-mp.M-diag+1; offset++ {
			c += df[offset]*dg[offset+diag] + df[offset+diag]*dg[offset]
			c_cmp = c * (sig[offset] * sig[offset+diag])
			if c_cmp > mpr.MP[offset] {
				if c_cmp > 1 {
					c_cmp = 1
				}
				mpr.MP[offset] = c_cmp
				mpr.Idx[offset] = offset + diag
			}
			if c_cmp > mpr.MP[offset+diag] {
				if c_cmp > 1 {
					c_cmp = 1
				}
				mpr.MP[offset+diag] = c_cmp
				mpr.Idx[offset+diag] = offset
			}
		}
	}

	for i := 0; i < len(mpr.MP); i++ {
		mpr.MP[i] = math.Sqrt(2 * float64(mp.M) * (1 - mpr.MP[i]))
	}

	return mpr
}

// mergeMPResults reads from a slice of channels for Matrix Profile results and
// updates the matrix profile in the struct
func (mp *MatrixProfile) mergeMPResults(results []chan *mpResult) error {
	var err error

	resultSlice := make([]*mpResult, len(results))
	for i := 0; i < len(results); i++ {
		resultSlice[i] = <-results[i]

		// if an error is encountered set the variable so that it can be checked
		// for at the end of processing. Tracks the last error emitted by any
		// batch
		if resultSlice[i].Err != nil {
			err = resultSlice[i].Err
			continue
		}

		// continues to the next loop if the result returned is empty but
		// had no errors
		if resultSlice[i].MP == nil || resultSlice[i].Idx == nil {
			continue
		}
		for j := 0; j < len(resultSlice[i].MP); j++ {
			if resultSlice[i].MP[j] <= mp.MP[j] {
				mp.MP[j] = resultSlice[i].MP[j]
				mp.Idx[j] = resultSlice[i].Idx[j]
			}
		}
	}
	return err
}

func (mp *MatrixProfile) mpx(parallelism int) error {
	if !mp.SelfJoin {
		return errors.New("MPX method only permits self joins")
	}
	diagMax := mp.N - mp.M + 1
	batchSize := (diagMax-mp.M/4+1)/parallelism + 1
	results := make([]chan *mpResult, parallelism)
	for i := 0; i < parallelism; i++ {
		results[i] = make(chan *mpResult)
	}

	// go routine to continually check for results on the slice of channels
	// for each batch kicked off. This merges the results of the batched go
	// routines by picking the lowest value in each batch's matrix profile and
	// updating the matrix profile index.
	var err error
	done := make(chan bool)
	go func() {
		err = mp.mergeMPResults(results)
		done <- true
	}()

	mu, sig := util.MuInvN(mp.A, mp.M)

	df := make([]float64, diagMax)
	dg := make([]float64, diagMax)
	for i := 0; i < len(df)-1; i++ {
		df[i+1] = 0.5 * (mp.A[mp.M+i] - mp.A[i])
		dg[i+1] = (mp.A[mp.M+i] - mu[1+i]) + (mp.A[i] - mu[i])
	}

	// kick off multiple go routines to process a batch of rows returning back
	// the matrix profile for that batch and any error encountered
	var wg sync.WaitGroup
	wg.Add(parallelism)
	for batch := 0; batch < parallelism; batch++ {
		go func(idx int) {
			results[idx] <- mp.mpxBatch(idx, mu, sig, df, dg, batchSize, &wg)
		}(batch)
	}
	wg.Wait()

	// waits for all results to be read and merged before returning success
	<-done

	return err
}

type Options struct {
	Method      method.Method
	Sample      float64
	Parallelism int

	KMotifs        int     // the top k motifs to find
	RMotifs        float64 // the max radius to find motifs
	KDiscords      int     // the top k discords to find
	OutputFilename string  // relative or absolute filepath for the visualization output
}

func NewOptions() Options {
	return Options{
		Method:         method.STOMP,
		Parallelism:    4,
		KMotifs:        3,
		RMotifs:        2.0,
		KDiscords:      3,
		OutputFilename: "mp.png",
	}
}

// Compute calculate the matrixprofile given a set of input options. This defaults to using
// STOMP unless specified differently
func (mp *MatrixProfile) Compute(opt Options) error {
	switch opt.Method {
	case method.STOMP:
		return mp.stomp(opt.Parallelism)
	case method.STAMP:
		return mp.stamp(opt.Sample, opt.Parallelism)
	case method.STMP:
		return mp.stmp()
	case method.MPX:
		return mp.mpx(opt.Parallelism)
	}
	return nil
}

// MotifGroup stores a list of indices representing a similar motif along
// with the minimum distance that this set of motif composes of.
type MotifGroup struct {
	Idx     []int
	MinDist float64
}

// TopKMotifs will iteratively go through the matrix profile to find the
// top k motifs with a given radius. Only applies to self joins.
func (mp MatrixProfile) TopKMotifs(k int, r float64) ([]MotifGroup, error) {
	if !mp.SelfJoin {
		return nil, errors.New("can only find top motifs if a self join is performed")
	}
	var err error
	var minDistIdx int

	motifs := make([]MotifGroup, k)

	mpCurrent, err := mp.applyAV()
	if err != nil {
		return nil, err
	}

	prof := make([]float64, len(mpCurrent)) // stores minimum matrix profile distance between motif pairs
	fft := fourier.NewFFT(mp.N)
	var j int

	for j = 0; j < k; j++ {
		// find minimum distance and index location
		motifDistance := math.Inf(1)
		minIdx := math.MaxInt64
		for i, d := range mpCurrent {
			if d < motifDistance {
				motifDistance = d
				minIdx = i
			}
		}

		if minIdx == math.MaxInt64 {
			// can't find any more motifs so returning what we currently found
			return motifs, nil
		}

		// filter out all indexes that have a distance within r*motifDistance
		motifSet := make(map[int]struct{})
		initialMotif := []int{minIdx, mp.Idx[minIdx]}
		motifSet[minIdx] = struct{}{}
		motifSet[mp.Idx[minIdx]] = struct{}{}

		if err = mp.distanceProfile(initialMotif[0], prof, fft); err != nil {
			return nil, err
		}

		// kill off any indices around the initial motif pair since they are
		// trivial solutions
		util.ApplyExclusionZone(prof, initialMotif[0], mp.M/2)
		util.ApplyExclusionZone(prof, initialMotif[1], mp.M/2)
		if j > 0 {
			for k := j; k >= 0; k-- {
				for _, idx := range motifs[k].Idx {
					util.ApplyExclusionZone(prof, idx, mp.M/2)
				}
			}
		}
		// keep looking for the closest index to the current motif. Each
		// index found will have an exclusion zone applied as to remove
		// trivial solutions. This eventually exits when there's nothing
		// found within the radius distance.
		for {
			minDistIdx = floats.MinIdx(prof)

			if prof[minDistIdx] < motifDistance*r {
				motifSet[minDistIdx] = struct{}{}
				util.ApplyExclusionZone(prof, minDistIdx, mp.M/2)
			} else {
				// the closest distance in the profile is greater than the desired
				// distance so break
				break
			}
		}

		// store the found motif indexes and create an exclusion zone around
		// each index in the current matrix profile
		motifs[j] = MotifGroup{
			Idx:     make([]int, 0, len(motifSet)),
			MinDist: motifDistance,
		}
		for idx := range motifSet {
			motifs[j].Idx = append(motifs[j].Idx, idx)
			util.ApplyExclusionZone(mpCurrent, idx, mp.M/2)
		}

		// sorts the indices in ascending order
		sort.IntSlice(motifs[j].Idx).Sort()
	}

	return motifs[:j], nil
}

// TopKDiscords finds the top k time series discords starting indexes from a computed
// matrix profile. Each discovery of a discord will apply an exclusion zone around
// the found index so that new discords can be discovered.
func (mp MatrixProfile) TopKDiscords(k int, exclusionZone int) ([]int, error) {
	mpCurrent, err := mp.applyAV()
	if err != nil {
		return nil, err
	}

	// if requested k is larger than length of the matrix profile, cap it
	if k > len(mpCurrent) {
		k = len(mpCurrent)
	}

	discords := make([]int, k)
	var maxVal float64
	var maxIdx int
	var i int

	for i = 0; i < k; i++ {
		maxVal = 0
		maxIdx = math.MaxInt64
		for j, val := range mpCurrent {
			if !math.IsInf(val, 1) && val > maxVal {
				maxVal = val
				maxIdx = j
			}
		}

		if maxIdx == math.MaxInt64 {
			break
		}

		discords[i] = maxIdx
		util.ApplyExclusionZone(mpCurrent, maxIdx, exclusionZone)
	}
	return discords[:i], nil
}

// Segment finds the the index where there may be a potential timeseries
// change. Returns the index of the potential change, value of the corrected
// arc curve score and the histogram of all the crossings for each index in
// the matrix profile index. This approach is based on the UCR paper on
// segmentation of timeseries using matrix profiles which can be found
// https://www.cs.ucr.edu/%7Eeamonn/Segmentation_ICDM.pdf
func (mp MatrixProfile) Segment() (int, float64, []float64) {
	histo := util.ArcCurve(mp.Idx)

	for i := 0; i < len(histo); i++ {
		if i == 0 || i == len(histo)-1 {
			histo[i] = math.Min(1.0, float64(len(histo)))
		} else {
			histo[i] = math.Min(1.0, histo[i]/util.Iac(float64(i), len(histo)))
		}
	}

	minIdx := math.MaxInt64
	minVal := math.Inf(1)
	for i := 0; i < len(histo); i++ {
		if histo[i] < minVal {
			minIdx = i
			minVal = histo[i]
		}
	}

	return minIdx, float64(minVal), histo
}

// applyAV applies an annotation vector to the current matrix profile. Annotation vector
// values must be between 0 and 1.
func (mp MatrixProfile) applyAV() ([]float64, error) {
	var avec []float64
	switch mp.AV {
	case av.Default:
		avec = av.MakeDefault(mp.B, mp.M)
	case av.Complexity:
		avec = av.MakeCompexity(mp.B, mp.M)
	case av.MeanStd:
		avec = av.MakeMeanStd(mp.B, mp.M)
	case av.Clipping:
		avec = av.MakeClipping(mp.B, mp.M)
	default:
		return nil, fmt.Errorf("invalid annotation vector specified with matrix profile, %d", mp.AV)
	}

	if len(avec) != len(mp.MP) {
		return nil, fmt.Errorf("annotation vector length, %d, does not match matrix profile length, %d", len(avec), len(mp.MP))
	}

	// find the maximum matrix profile value
	maxMP := 0.0
	for _, val := range mp.MP {
		if val > maxMP {
			maxMP = val
		}
	}

	// check that all annotation vector values are between 0 and 1
	for idx, val := range avec {
		if val < 0.0 || val > 1.0 {
			return nil, fmt.Errorf("got an annotation vector value of %.3f at index %d. must be between 0 and 1", val, idx)
		}
	}

	// applies the matrix profile correction. 1 results in no change to the matrix profile and
	// 0 results in lifting the current matrix profile value by the maximum matrix profile value
	out := make([]float64, len(mp.MP))
	for idx, val := range avec {
		out[idx] = mp.MP[idx] + (1-val)*maxMP
	}

	return out, nil
}

// Visualize creates a png of the matrix profile given a matrix profile.
func (mp MatrixProfile) Visualize(fn string, motifs []MotifGroup, discords []int, cac []float64) error {
	sigPts := Points(mp.A, len(mp.A))
	mpPts := Points(mp.MP, len(mp.A))
	cacPts := Points(cac, len(mp.A))
	motifPts := make([][]plotter.XYs, len(motifs))
	discordPts := make([]plotter.XYs, len(discords))
	discordLabels := make([]string, len(discords))

	for i := 0; i < len(motifs); i++ {
		motifPts[i] = make([]plotter.XYs, len(motifs[i].Idx))
	}

	for i := 0; i < len(motifs); i++ {
		for j, idx := range motifs[i].Idx {
			motifPts[i][j] = points(mp.A[idx:idx+mp.M], mp.M)
		}
	}

	for i, idx := range discords {
		discordPts[i] = points(mp.A[idx:idx+mp.M], mp.M)
		discordLabels[i] = strconv.Itoa(idx)
	}

	return plotMP(sigPts, mpPts, cacPts, motifPts, discordPts, discordLabels, fn)
}

func points(a []float64, n int) plotter.XYs {
	pts := make(plotter.XYs, n)
	for i := 0; i < n; i++ {
		pts[i].X = float64(i)
		if i < len(a) {
			pts[i].Y = a[i]
		}
	}
	return pts
}

func createPlot(pts []plotter.XYs, labels []string, title string) (*plot.Plot, error) {
	if labels != nil && len(pts) != len(labels) {
		return nil, fmt.Errorf("number of XYs, %d, does not match number of labels, %d", len(pts), len(labels))
	}

	p, err := plot.New()
	if err != nil {
		return p, err
	}

	p.Title.Text = title
	for i := 0; i < len(pts); i++ {
		line, points, err := plotter.NewLinePoints(pts[i])
		if err != nil {
			return p, err
		}
		line.Color = plotutil.Color(i)
		points.Color = plotutil.Color(i)
		points.Shape = nil
		p.Add(line, points)
		if labels != nil {
			p.Legend.Add(labels[i], line)
		}
	}
	return p, err
}

func plotMP(sigPts, mpPts, cacPts plotter.XYs, motifPts [][]plotter.XYs, discordPts []plotter.XYs, discordLabels []string, filename string) error {
	var err error
	rows, cols := len(motifPts), 2
	if rows < 4 {
		rows = 4
	}
	plots := make([][]*plot.Plot, rows)

	for i := 0; i < len(motifPts); i++ {
		plots[i] = make([]*plot.Plot, cols)
	}

	plots[0][0], err = createPlot([]plotter.XYs{sigPts}, nil, "signal")
	if err != nil {
		return err
	}

	plots[1][0], err = createPlot([]plotter.XYs{mpPts}, nil, "matrix profile")
	if err != nil {
		return err
	}

	plots[2][0], err = createPlot([]plotter.XYs{cacPts}, nil, "corrected arc curve")
	if err != nil {
		return err
	}

	plots[3][0], err = createPlot(discordPts, discordLabels, "discords")
	if err != nil {
		return err
	}

	for i := 0; i < len(motifPts); i++ {
		plots[i][1], err = createPlot(motifPts[i], nil, fmt.Sprintf("motif %d", i))
		if err != nil {
			return err
		}
	}

	img := vgimg.New(vg.Points(1200), vg.Points(600))
	dc := draw.New(img)

	t := draw.Tiles{
		Rows: rows,
		Cols: cols,
	}

	canvases := plot.Align(plots, t, dc)
	for j := 0; j < rows; j++ {
		for i := 0; i < cols; i++ {
			if plots[j][i] != nil {
				plots[j][i].Draw(canvases[j][i])
			}
		}
	}

	w, err := os.Create(filename)
	if err != nil {
		return err
	}

	png := vgimg.PngCanvas{Canvas: img}
	_, err = png.WriteTo(w)
	return err
}

// Analyze performs the matrix profile computation and discovers various features
// from the profile such as motifs, discords, and segmentation. The results are
// visualized and saved into an output file.
func (mp MatrixProfile) Analyze(o Options) error {
	var err error
	if err = mp.Compute(o); err != nil {
		return err
	}

	_, _, cac := mp.Segment()

	motifs, err := mp.TopKMotifs(o.KMotifs, o.RMotifs)
	if err != nil {
		return err
	}

	discords, err := mp.TopKDiscords(o.KDiscords, mp.M/2)
	if err != nil {
		return err
	}

	return mp.Visualize(o.OutputFilename, motifs, discords, cac)
}
