// Package matrixprofile computes the matrix profile and matrix profile index of a time series
package matrixprofile

import (
	"container/heap"
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"math"
	"math/rand"
	"os"
	"runtime"
	"sort"
	"strconv"
	"sync"

	"github.com/matrix-profile-foundation/go-matrixprofile/av"
	"github.com/matrix-profile-foundation/go-matrixprofile/util"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/fourier"
	"gonum.org/v1/plot/plotter"
)

// MatrixProfile is a struct that tracks the current matrix profile computation
// for a given timeseries of length N and subsequence length of W. The profile
// and the profile index are stored here.
type MatrixProfile struct {
	A        []float64    `json:"a"`                 // query time series
	B        []float64    `json:"b"`                 // timeseries to perform full join with
	AMean    []float64    `json:"a_mean"`            // sliding mean of a with a window of m each
	AStd     []float64    `json:"a_std"`             // sliding standard deviation of a with a window of m each
	BMean    []float64    `json:"b_mean"`            // sliding mean of b with a window of m each
	BStd     []float64    `json:"b_std"`             // sliding standard deviation of b with a window of m each
	BF       []complex128 `json:"b_fft"`             // holds an existing calculation of the FFT of b timeseries
	N        int          `json:"n"`                 // length of the timeseries
	W        int          `json:"w"`                 // length of a subsequence
	SelfJoin bool         `json:"self_join"`         // indicates whether a self join is performed with an exclusion zone
	MP       []float64    `json:"mp"`                // matrix profile
	Idx      []int        `json:"pi"`                // matrix profile index
	MPB      []float64    `json:"mp_ba"`             // matrix profile for the BA join
	IdxB     []int        `json:"pi_ba"`             // matrix profile index for the BA join
	AV       av.AV        `json:"annotation_vector"` // type of annotation vector which defaults to all ones
	Opts     *MPOpts      `json:"options"`           // options used for the computation
}

// New creates a matrix profile struct with a given timeseries length n and
// subsequence length of m. The first slice, a, is used as the initial
// timeseries to join with the second, b. If b is nil, then the matrix profile
// assumes a self join on the first timeseries.
func New(a, b []float64, w int) (*MatrixProfile, error) {
	if a == nil || len(a) == 0 {
		return nil, fmt.Errorf("first slice is nil or has a length of 0")
	}

	if b != nil && len(b) == 0 {
		return nil, fmt.Errorf("second slice must be nil for self-join operation or have a length greater than 0")
	}

	mp := MatrixProfile{
		A: a,
		W: w,
		N: len(b),
	}
	if b == nil {
		mp.N = len(a)
		mp.B = a
		mp.SelfJoin = true
	} else {
		mp.B = b
	}

	if mp.W > len(mp.A) || mp.W > len(mp.B) {
		return nil, fmt.Errorf("subsequence length must be less than the timeseries")
	}

	if mp.W < 2 {
		return nil, fmt.Errorf("subsequence length must be at least 2")
	}

	mp.AV = av.Default

	return &mp, nil
}

func applySingleAV(mp, ts []float64, w int, a av.AV) ([]float64, error) {
	avec, err := av.Create(a, ts, w)
	if err != nil {
		return nil, err
	}

	if len(avec) != len(mp) {
		return nil, fmt.Errorf("annotation vector length, %d, does not match matrix profile length, %d", len(avec), len(mp))
	}

	// find the maximum matrix profile value
	maxMP := 0.0
	for _, val := range mp {
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
	out := make([]float64, len(mp))
	for idx, val := range avec {
		out[idx] = mp[idx] + (1-val)*maxMP
	}

	return out, nil
}

// ApplyAV applies an annotation vector to the current matrix profile. Annotation vector
// values must be between 0 and 1.
func (mp MatrixProfile) ApplyAV() ([]float64, []float64, error) {
	var err error
	abmp := make([]float64, len(mp.MP))
	bamp := make([]float64, len(mp.MPB))

	copy(abmp, mp.MP)
	copy(bamp, mp.MPB)
	if !mp.Opts.Euclidean {
		util.P2E(abmp, mp.W)
		util.P2E(bamp, mp.W)
	}

	abmp, err = applySingleAV(abmp, mp.A, mp.W, mp.AV)
	if err != nil {
		return nil, nil, err
	}

	if mp.MPB != nil {
		bamp, err = applySingleAV(bamp, mp.B, mp.W, mp.AV)
	}

	if err != nil {
		return nil, nil, err
	}

	if !mp.Opts.Euclidean {
		util.E2P(abmp, mp.W)
		util.E2P(bamp, mp.W)
	}

	return abmp, bamp, nil
}

// Save will save the current matrix profile struct to disk
func (mp MatrixProfile) Save(filepath, format string) error {
	var err error
	switch format {
	case "json":
		f, err := os.Open(filepath)
		if err != nil {
			f, err = os.Create(filepath)
			if err != nil {
				return err
			}
		}
		defer f.Close()
		out, err := json.Marshal(mp)
		if err != nil {
			return err
		}
		_, err = f.Write(out)
	default:
		return fmt.Errorf("invalid save format, %s", format)
	}
	return err
}

// Load will attempt to load a matrix profile from a file for iterative use
func (mp *MatrixProfile) Load(filepath, format string) error {
	var err error
	switch format {
	case "json":
		f, err := os.Open(filepath)
		if err != nil {
			return err
		}
		defer f.Close()
		b, err := ioutil.ReadAll(f)
		if err != nil {
			return err
		}
		err = json.Unmarshal(b, mp)
	default:
		return fmt.Errorf("invalid load format, %s", format)
	}
	return err
}

type mpVals []float64

func (m mpVals) Len() int {
	return len(m)
}

func (m mpVals) Swap(i, j int) {
	m[i], m[j] = m[j], m[i]
}

func (m mpVals) Less(i, j int) bool {
	return m[i] < m[j]
}

// Push implements the function in the heap interface
func (m *mpVals) Push(x interface{}) {
	*m = append(*m, x.(float64))
}

// Pop implements the function in the heap interface
func (m *mpVals) Pop() interface{} {
	x := (*m)[len(*m)-1]
	*m = (*m)[:len(*m)-1]
	return x
}

type MPDistOpts struct {
	AV   av.AV
	Opts *MPOpts
}

func NewMPDistOpts() *MPDistOpts {
	return &MPDistOpts{
		AV:   av.Default,
		Opts: NewMPOpts(),
	}
}

// MPDist computes the matrix profile distance measure between a and b with a
// subsequence window of m.
func MPDist(a, b []float64, w int, o *MPDistOpts) (float64, error) {
	if o == nil {
		o = NewMPDistOpts()
	}

	mp, err := New(a, b, w)
	if err != nil {
		return 0, err
	}

	if err = mp.Compute(o.Opts); err != nil {
		return 0, nil
	}

	mpab, mpba, err := mp.ApplyAV()
	if err != nil {
		return 0, nil
	}

	thresh := 0.05
	k := int(thresh * float64(len(a)+len(b)))
	mpABBASize := len(mpab) + len(mpba)

	if k < mpABBASize {
		var lowestMPs mpVals
		heap.Init(&lowestMPs)
		for _, d := range mpab {
			// since this is a max heap and correlations go from 0-1 we need high correlations
			// to stay in the heap with the poorest correlation at the root.
			if !mp.Opts.Euclidean {
				d = -d
			}
			if len(lowestMPs) == k+1 {
				if d < lowestMPs[0] {
					heap.Pop(&lowestMPs)
					heap.Push(&lowestMPs, d)
				}
			} else {
				heap.Push(&lowestMPs, d)
			}
		}

		for _, d := range mpba {
			// since this is a max heap and correlations go from 0-1 we need high correlations
			// to stay in the heap with the poorest correlation at the root.
			if !mp.Opts.Euclidean {
				d = -d
			}

			if len(lowestMPs) == k+1 {
				if d < lowestMPs[0] {
					heap.Pop(&lowestMPs)
					heap.Push(&lowestMPs, d)
				}
			} else {
				heap.Push(&lowestMPs, d)
			}
		}

		if !mp.Opts.Euclidean {
			return -lowestMPs[0], nil
		}
		return lowestMPs[0], nil
	}

	var trackVal float64
	if !mp.Opts.Euclidean {
		trackVal = 1
	}

	for _, d := range mp.MP {
		if mp.Opts.Euclidean {
			if d > trackVal {
				trackVal = d
			}
		} else {
			if d < trackVal {
				trackVal = d
			}
		}
	}

	for _, d := range mp.MPB {
		if mp.Opts.Euclidean {
			if d > trackVal {
				trackVal = d
			}
		} else {
			if d < trackVal {
				trackVal = d
			}
		}
	}

	return trackVal, nil
}

type Algo string

const (
	AlgoSTOMP Algo = "stomp"
	AlgoSTAMP Algo = "stamp"
	AlgoSTMP  Algo = "stmp"
	AlgoMPX   Algo = "mpx"
)

// MPOpts are parameters to vary the algorithm to compute the matrix profile.
type MPOpts struct {
	Algorithm    Algo    `json:"algorithm"`  // choose which algorithm to compute the matrix profile
	Sample       float64 `json:"sample_pct"` // only applicable to algorithm STAMP
	Parallelism  int     `json:"parallelism"`
	Euclidean    bool    `json:"euclidean"`                  // defaults to using euclidean distance instead of pearson correlation for matrix profile
	RemapNegCorr bool    `json:"remap_negative_correlation"` // defaults to no remapping. This is used so that highly negatively correlated sequences will show a low distance as well.
}

// NewMPOpts returns a default MPOpts
func NewMPOpts() *MPOpts {
	p := runtime.NumCPU() * 2
	if p < 1 {
		p = 1
	}
	return &MPOpts{
		Algorithm:   AlgoMPX,
		Sample:      1.0,
		Parallelism: p,
		Euclidean:   true,
	}
}

// Compute calculate the matrixprofile given a set of input options.
func (mp *MatrixProfile) Compute(o *MPOpts) error {
	if o == nil {
		o = NewMPOpts()
	}
	mp.Opts = o

	switch o.Algorithm {
	case AlgoSTOMP:
		return mp.stomp()
	case AlgoSTAMP:
		return mp.stamp()
	case AlgoSTMP:
		return mp.stmp()
	case AlgoMPX:
		return mp.mpx()
	}
	return nil
}

// initCaches initializes cached data including the timeseries a and b rolling mean
// and standard deviation and full fourier transform of timeseries b
func (mp *MatrixProfile) initCaches() error {
	var err error
	// precompute the mean and standard deviation for each window of size m for all
	// sliding windows across the b timeseries
	mp.BMean, mp.BStd, err = util.MovMeanStd(mp.B, mp.W)
	if err != nil {
		return err
	}

	mp.AMean, mp.AStd, err = util.MovMeanStd(mp.A, mp.W)
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
		qpad[i] = q[mp.W-i-1]
	}
	qf := fft.Coefficients(nil, qpad)

	// in place multiply the fourier transform of the b time series with
	// the subsequence fourier transform and store in the subsequence fft slice
	for i := 0; i < len(qf); i++ {
		qf[i] = mp.BF[i] * qf[i]
	}

	dot := fft.Sequence(nil, qf)

	for i := 0; i < mp.N-mp.W+1; i++ {
		dot[mp.W-1+i] = dot[mp.W-1+i] / float64(mp.N)
	}
	return dot[mp.W-1:]
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
		profile[i] = math.Sqrt(math.Abs(2 * (float64(mp.W) - (dot[i] / mp.BStd[i]))))
	}
	return nil
}

// distanceProfile computes the distance profile between a and b time series.
// If b is set to nil then it assumes a self join and will create an exclusion
// area for trivial nearest neighbors. Writes the euclidean distance between
// the specified subsequence in mp.A with each subsequence in mp.B to profile
func (mp MatrixProfile) distanceProfile(idx int, profile []float64, fft *fourier.FFT) error {
	if idx > len(mp.A)-mp.W {
		return fmt.Errorf("provided index  %d is beyond the length of timeseries %d minus the subsequence length %d", idx, len(mp.A), mp.W)
	}

	if err := mp.mass(mp.A[idx:idx+mp.W], profile, fft); err != nil {
		return err
	}

	// sets the distance in the exclusion zone to +Inf
	if mp.SelfJoin {
		util.ApplyExclusionZone(profile, idx, mp.W/2)
	}
	return nil
}

// calculateDistanceProfile converts a sliding dot product slice of floats into
// distances and normalizes the output. Writes results back into the profile slice
// of floats representing the distance profile.
func (mp MatrixProfile) calculateDistanceProfile(dot []float64, idx int, profile []float64) error {
	if idx > len(mp.A)-mp.W {
		return fmt.Errorf("provided index %d is beyond the length of timeseries a %d minus the subsequence length %d", idx, len(mp.A), mp.W)
	}

	if len(profile) != len(dot) {
		return fmt.Errorf("profile length, %d, is not the same as the dot product length, %d", len(profile), len(dot))
	}

	// converting cross correlation value to euclidian distance
	for i := 0; i < len(dot); i++ {
		profile[i] = math.Sqrt(2 * float64(mp.W) * math.Abs(1-(dot[i]-float64(mp.W)*mp.BMean[i]*mp.AMean[idx])/(float64(mp.W)*mp.BStd[i]*mp.AStd[idx])))
	}

	if mp.SelfJoin {
		// sets the distance in the exclusion zone to +Inf
		util.ApplyExclusionZone(profile, idx, mp.W/2)
	}
	return nil
}

// stmp computes the full matrix profile given two time series as inputs.
// If the second time series is set to nil then a self join on the first
// will be performed. Stores the matrix profile and matrix profile index
// in the struct.
func (mp *MatrixProfile) stmp() error {
	if err := mp.initCaches(); err != nil {
		return err
	}

	mp.MP = make([]float64, mp.N-mp.W+1)
	mp.Idx = make([]int, mp.N-mp.W+1)
	for i := 0; i < len(mp.MP); i++ {
		mp.MP[i] = math.Inf(1)
		mp.Idx[i] = math.MaxInt64
	}

	var err error
	profile := make([]float64, mp.N-mp.W+1)

	fft := fourier.NewFFT(mp.N)
	for i := 0; i < mp.N-mp.W+1; i++ {
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
		if err = mp.distanceProfile(len(mp.A)-mp.W, profile, fft); err != nil {
			return err
		}

		minVal := math.Inf(1)
		minIdx := math.MaxInt64
		for j := 0; j < len(profile)-1; j++ {
			if profile[j] <= mp.MP[j] {
				mp.MP[j] = profile[j]
				mp.Idx[j] = mp.N - mp.W
			}
			if profile[j] < minVal {
				minVal = profile[j]
				minIdx = j
			}
		}
		mp.MP[mp.N-mp.W] = minVal
		mp.Idx[mp.N-mp.W] = minIdx
	}
	return nil
}

// mpResult is the output struct from a batch processing for STAMP, STOMP, and MPX. This struct
// can later be merged together in linear time or with a divide and conquer approach
type mpResult struct {
	MP   []float64
	Idx  []int
	MPB  []float64
	IdxB []int
	Err  error
}

// mergeMPResults reads from a slice of channels for Matrix Profile results and
// updates the matrix profile in the struct
func (mp *MatrixProfile) mergeMPResults(results []chan *mpResult, euclidean bool) error {
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
			if euclidean {
				if resultSlice[i].MP[j] <= mp.MP[j] {
					mp.MP[j] = resultSlice[i].MP[j]
					mp.Idx[j] = resultSlice[i].Idx[j]
				}
			} else {
				if math.Abs(resultSlice[i].MP[j]) < math.Abs(mp.MP[j]) {
					mp.MP[j] = resultSlice[i].MP[j]
					mp.Idx[j] = resultSlice[i].Idx[j]
				}
			}
		}

		// check if the BA join has results and merge if so
		if resultSlice[i].MPB == nil || resultSlice[i].IdxB == nil {
			continue
		}
		for j := 0; j < len(resultSlice[i].MPB); j++ {
			if euclidean {
				if resultSlice[i].MPB[j] <= mp.MPB[j] {
					mp.MPB[j] = resultSlice[i].MPB[j]
					mp.IdxB[j] = resultSlice[i].IdxB[j]
				}
			} else {
				if math.Abs(resultSlice[i].MPB[j]) < math.Abs(mp.MPB[j]) {
					mp.MPB[j] = resultSlice[i].MPB[j]
					mp.IdxB[j] = resultSlice[i].IdxB[j]
				}
			}
		}

	}
	return err
}

// stamp uses random ordering to compute the matrix profile. User can specify the
// sample to be anything between 0 and 1 so that the computation early terminates
// and provides the current computed matrix profile. 1 represents the exact matrix
// profile. This should compute far faster at the cost of an approximation of the
// matrix profile. Stores the matrix profile and matrix profile index in the struct.
func (mp *MatrixProfile) stamp() error {
	if mp.Opts.Sample <= 0.0 {
		return fmt.Errorf("must provide a sampling greater than 0 and at most 1, sample: %.3f", mp.Opts.Sample)
	}

	if err := mp.initCaches(); err != nil {
		return err
	}

	mp.MP = make([]float64, mp.N-mp.W+1)
	mp.Idx = make([]int, mp.N-mp.W+1)
	for i := 0; i < len(mp.MP); i++ {
		mp.MP[i] = math.Inf(1)
		mp.Idx[i] = math.MaxInt64
	}

	randIdx := rand.Perm(len(mp.A) - mp.W + 1)

	batchSize := (len(mp.A)-mp.W+1)/mp.Opts.Parallelism + 1
	results := make([]chan *mpResult, mp.Opts.Parallelism)
	for i := 0; i < mp.Opts.Parallelism; i++ {
		results[i] = make(chan *mpResult)
	}

	// go routine to continually check for results on the slice of channels
	// for each batch kicked off. This merges the results of the batched go
	// routines by picking the lowest value in each batch's matrix profile and
	// updating the matrix profile index.
	var err error
	done := make(chan bool)
	go func() {
		err = mp.mergeMPResults(results, true)
		done <- true
	}()

	// kick off multiple go routines to process a batch of rows returning back
	// the matrix profile for that batch and any error encountered
	var wg sync.WaitGroup
	wg.Add(mp.Opts.Parallelism)
	for batch := 0; batch < mp.Opts.Parallelism; batch++ {
		go func(idx int) {
			results[idx] <- mp.stampBatch(idx, batchSize, mp.Opts.Sample, randIdx, &wg)
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
	if idx*batchSize+mp.W > len(mp.A) {
		// got an index larger than mp.A so ignore
		return &mpResult{}
	}

	// initialize this batch's matrix profile results
	result := &mpResult{
		MP:  make([]float64, mp.N-mp.W+1),
		Idx: make([]int, mp.N-mp.W+1),
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
			return &mpResult{nil, nil, nil, nil, err}
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

// stomp is an optimization on the STAMP approach reducing the runtime from O(n^2logn)
// down to O(n^2). This is an ordered approach, since the sliding dot product or cross
// correlation can be easily updated for the next sliding window, if the previous window
// dot product is available. This should also greatly reduce the number of memory
// allocations needed to compute an arbitrary timeseries length.
func (mp *MatrixProfile) stomp() error {
	if err := mp.initCaches(); err != nil {
		return err
	}

	mp.MP = make([]float64, mp.N-mp.W+1)
	mp.Idx = make([]int, mp.N-mp.W+1)
	for i := 0; i < len(mp.MP); i++ {
		mp.MP[i] = math.Inf(1)
		mp.Idx[i] = math.MaxInt64
	}

	batchSize := (len(mp.A)-mp.W+1)/mp.Opts.Parallelism + 1
	results := make([]chan *mpResult, mp.Opts.Parallelism)
	for i := 0; i < mp.Opts.Parallelism; i++ {
		results[i] = make(chan *mpResult)
	}

	// go routine to continually check for results on the slice of channels
	// for each batch kicked off. This merges the results of the batched go
	// routines by picking the lowest value in each batch's matrix profile and
	// updating the matrix profile index.
	var err error
	done := make(chan bool)
	go func() {
		err = mp.mergeMPResults(results, true)
		done <- true
	}()

	// kick off multiple go routines to process a batch of rows returning back
	// the matrix profile for that batch and any error encountered
	var wg sync.WaitGroup
	wg.Add(mp.Opts.Parallelism)
	for batch := 0; batch < mp.Opts.Parallelism; batch++ {
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
	if idx*batchSize+mp.W > len(mp.A) {
		// got an index larger than mp.A so ignore
		return &mpResult{}
	}

	// compute for this batch the first row's sliding dot product
	fft := fourier.NewFFT(mp.N)
	dot := mp.crossCorrelate(mp.A[idx*batchSize:idx*batchSize+mp.W], fft)

	profile := make([]float64, len(dot))
	var err error
	if err = mp.calculateDistanceProfile(dot, idx*batchSize, profile); err != nil {
		return &mpResult{nil, nil, nil, nil, err}
	}

	// initialize this batch's matrix profile results
	result := &mpResult{
		MP:  make([]float64, mp.N-mp.W+1),
		Idx: make([]int, mp.N-mp.W+1),
	}

	copy(result.MP, profile)
	for i := 0; i < len(profile); i++ {
		result.Idx[i] = idx * batchSize
	}

	// iteratively update for this batch each row's matrix profile and matrix
	// profile index
	var nextDotZero float64
	for i := 1; i < batchSize; i++ {
		if idx*batchSize+i-1 >= len(mp.A) || idx*batchSize+i+mp.W-1 >= len(mp.A) {
			// looking for an index beyond the length of mp.A so ignore and move one
			// with the current processed matrix profile
			break
		}
		for j := mp.N - mp.W; j > 0; j-- {
			dot[j] = dot[j-1] - mp.B[j-1]*mp.A[idx*batchSize+i-1] + mp.B[j+mp.W-1]*mp.A[idx*batchSize+i+mp.W-1]
		}

		// recompute the first cross correlation since the algorithm is only valid for
		// points after it. Previous optimization of using a precomputed cache ONLY applies
		// if we're doing a self-join and is invalidated with AB-joins of different time series
		nextDotZero = 0
		for k := 0; k < mp.W; k++ {
			nextDotZero += mp.A[idx*batchSize+i+k] * mp.B[k]
		}
		dot[0] = nextDotZero
		if err = mp.calculateDistanceProfile(dot, idx*batchSize+i, profile); err != nil {
			return &mpResult{nil, nil, nil, nil, err}
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

func (mp *MatrixProfile) mpx() error {
	lenA := len(mp.A) - mp.W + 1
	lenB := len(mp.B) - mp.W + 1

	mp.MP = make([]float64, lenA)
	mp.Idx = make([]int, lenA)
	for i := 0; i < len(mp.MP); i++ {
		mp.MP[i] = math.Inf(1)
		mp.Idx[i] = math.MaxInt64
	}

	if !mp.SelfJoin {
		mp.MPB = make([]float64, lenB)
		mp.IdxB = make([]int, lenB)
		for i := 0; i < len(mp.MPB); i++ {
			mp.MPB[i] = math.Inf(1)
			mp.IdxB[i] = math.MaxInt64
		}
	}

	mua, siga := util.MuInvN(mp.A, mp.W)
	mub, sigb := mua, siga
	if !mp.SelfJoin {
		mub, sigb = util.MuInvN(mp.B, mp.W)
	}

	dfa := make([]float64, lenA)
	dga := make([]float64, lenA)
	for i := 0; i < lenA-1; i++ {
		dfa[i+1] = 0.5 * (mp.A[mp.W+i] - mp.A[i])
		dga[i+1] = (mp.A[mp.W+i] - mua[1+i]) + (mp.A[i] - mua[i])
	}

	dfb, dgb := dfa, dga
	if !mp.SelfJoin {
		dfb = make([]float64, lenB)
		dgb = make([]float64, lenB)
		for i := 0; i < lenB-1; i++ {
			dfb[i+1] = 0.5 * (mp.B[mp.W+i] - mp.B[i])
			dgb[i+1] = (mp.B[mp.W+i] - mub[1+i]) + (mp.B[i] - mub[i])
		}
	}

	// setup for AB join
	batchScheme := util.DiagBatchingScheme(lenA, mp.Opts.Parallelism)
	results := make([]chan *mpResult, mp.Opts.Parallelism)
	for i := 0; i < mp.Opts.Parallelism; i++ {
		results[i] = make(chan *mpResult)
	}

	// go routine to continually check for results on the slice of channels
	// for each batch kicked off. This merges the results of the batched go
	// routines by picking the lowest value in each batch's matrix profile and
	// updating the matrix profile index.
	var err error
	done := make(chan bool)
	go func() {
		err = mp.mergeMPResults(results, mp.Opts.Euclidean)
		done <- true
	}()

	// kick off multiple go routines to process a batch of rows returning back
	// the matrix profile for that batch and any error encountered
	var wg sync.WaitGroup
	wg.Add(mp.Opts.Parallelism)
	for batch := 0; batch < mp.Opts.Parallelism; batch++ {
		go func(batchNum int) {
			b := batchScheme[batchNum]
			if mp.SelfJoin {
				results[batchNum] <- mp.mpxBatch(b.Idx, mua, siga, dfa, dga, b.Size, &wg)
			} else {
				results[batchNum] <- mp.mpxabBatch(b.Idx, mua, siga, dfa, dga, mub, sigb, dfb, dgb, b.Size, &wg)
			}
		}(batch)
	}
	wg.Wait()

	// waits for all results to be read and merged before returning success
	<-done

	if mp.SelfJoin || err != nil {
		return err
	}

	// setup for BA join
	batchScheme = util.DiagBatchingScheme(lenB, mp.Opts.Parallelism)
	results = make([]chan *mpResult, mp.Opts.Parallelism)
	for i := 0; i < mp.Opts.Parallelism; i++ {
		results[i] = make(chan *mpResult)
	}

	// go routine to continually check for results on the slice of channels
	// for each batch kicked off. This merges the results of the batched go
	// routines by picking the lowest value in each batch's matrix profile and
	// updating the matrix profile index.
	go func() {
		err = mp.mergeMPResults(results, mp.Opts.Euclidean)
		done <- true
	}()

	// kick off multiple go routines to process a batch of rows returning back
	// the matrix profile for that batch and any error encountered
	wg.Add(mp.Opts.Parallelism)
	for batch := 0; batch < mp.Opts.Parallelism; batch++ {
		go func(batchNum int) {
			b := batchScheme[batchNum]
			results[batchNum] <- mp.mpxbaBatch(b.Idx, mua, siga, dfa, dga, mub, sigb, dfb, dgb, b.Size, &wg)
		}(batch)
	}
	wg.Wait()

	// waits for all results to be read and merged before returning success
	<-done

	return err
}

// mpxBatch processes a batch set of rows in matrix profile calculation.
func (mp MatrixProfile) mpxBatch(idx int, mu, sig, df, dg []float64, batchSize int, wg *sync.WaitGroup) *mpResult {
	defer wg.Done()
	exclZone := 1 // for seljoin we should at least get rid of neighboring points
	if mp.W/4 > exclZone {
		exclZone = mp.W / 4
	}
	if idx+exclZone > len(mp.A)-mp.W+1 {
		// got an index larger than max lag so ignore
		return &mpResult{}
	}

	mpr := &mpResult{
		MP:  make([]float64, len(mp.A)-mp.W+1),
		Idx: make([]int, len(mp.A)-mp.W+1),
	}
	for i := 0; i < len(mpr.MP); i++ {
		mpr.MP[i] = -1
	}

	var c, c_cmp float64
	s1 := make([]float64, mp.W)
	s2 := make([]float64, mp.W)
	for diag := idx + exclZone; diag < idx+batchSize+exclZone; diag++ {
		if diag >= len(mp.A)-mp.W+1 {
			break
		}

		//for i := 0; i < mp.W; i++ {
		//	c += (mp.A[diag+i] - mu[diag]) * (mp.A[i] - mu[0])
		//}
		copy(s1, mp.A[diag:diag+mp.W])
		copy(s2, mp.A[:mp.W])
		floats.AddConst(-mu[diag], s1)
		floats.AddConst(mu[0], s2)
		c = floats.Dot(s1, s2)

		for offset := 0; offset < len(mp.A)-mp.W-diag+1; offset++ {
			c += df[offset]*dg[offset+diag] + df[offset+diag]*dg[offset]
			c_cmp = c * (sig[offset] * sig[offset+diag])
			if mp.Opts.RemapNegCorr && c_cmp < 0 {
				c_cmp = -c_cmp
			}
			if c_cmp > mpr.MP[offset] {
				mpr.MP[offset] = c_cmp
				mpr.Idx[offset] = offset + diag
			}
			if c_cmp > mpr.MP[offset+diag] {
				mpr.MP[offset+diag] = c_cmp
				mpr.Idx[offset+diag] = offset
			}
		}
	}

	if mp.Opts.Euclidean {
		util.P2E(mpr.MP, mp.W)
	}

	return mpr
}

// mpxabBatch processes a batch set of rows in matrix profile AB join calculation.
func (mp MatrixProfile) mpxabBatch(idx int, mua, siga, dfa, dga, mub, sigb, dfb, dgb []float64, batchSize int, wg *sync.WaitGroup) *mpResult {
	defer wg.Done()
	lenA := len(mp.A) - mp.W + 1
	lenB := len(mp.B) - mp.W + 1

	if idx > lenA {
		// got an index larger than max lag so ignore
		return &mpResult{}
	}

	mpr := &mpResult{
		MP:   make([]float64, lenA),
		Idx:  make([]int, lenA),
		MPB:  make([]float64, lenB),
		IdxB: make([]int, lenB),
	}
	for i := 0; i < len(mpr.MP); i++ {
		mpr.MP[i] = -1
	}
	for i := 0; i < len(mpr.MPB); i++ {
		mpr.MPB[i] = -1
	}

	var c, c_cmp float64
	var offsetMax int
	s1 := make([]float64, mp.W)
	s2 := make([]float64, mp.W)
	for diag := idx; diag < idx+batchSize; diag++ {
		if diag >= lenA {
			break
		}

		//for i := 0; i < mp.W; i++ {
		//	c += (mp.A[diag+i] - mua[diag]) * (mp.B[i] - mub[0])
		//}
		copy(s1, mp.A[diag:diag+mp.W])
		copy(s2, mp.B[:mp.W])
		floats.AddConst(-mua[diag], s1)
		floats.AddConst(mub[0], s2)
		c = floats.Dot(s1, s2)

		offsetMax = lenA - diag
		if offsetMax > lenB {
			offsetMax = lenB
		}

		for offset := 0; offset < offsetMax; offset++ {
			c += dfb[offset]*dga[offset+diag] + dfa[offset+diag]*dgb[offset]
			c_cmp = c * (sigb[offset] * siga[offset+diag])
			if mp.Opts.RemapNegCorr && c_cmp < 0 {
				c_cmp = -c_cmp
			}
			if c_cmp > mpr.MP[offset+diag] {
				mpr.MP[offset+diag] = c_cmp
				mpr.Idx[offset+diag] = offset
			}
			if c_cmp > mpr.MPB[offset] {
				mpr.MPB[offset] = c_cmp
				mpr.IdxB[offset] = offset + diag
			}
		}
	}

	if mp.Opts.Euclidean {
		util.P2E(mpr.MP, mp.W)
		util.P2E(mpr.MPB, mp.W)
	}

	return mpr
}

// mpxbaBatch processes a batch set of rows in matrix profile calculation.
func (mp MatrixProfile) mpxbaBatch(idx int, mua, siga, dfa, dga, mub, sigb, dfb, dgb []float64, batchSize int, wg *sync.WaitGroup) *mpResult {
	defer wg.Done()
	lenA := len(mp.A) - mp.W + 1
	lenB := len(mp.B) - mp.W + 1

	if idx > lenA {
		// got an index larger than max lag so ignore
		return &mpResult{}
	}

	mpr := &mpResult{
		MP:   make([]float64, lenA),
		Idx:  make([]int, lenA),
		MPB:  make([]float64, lenB),
		IdxB: make([]int, lenB),
	}
	for i := 0; i < len(mpr.MP); i++ {
		mpr.MP[i] = -1
	}
	for i := 0; i < len(mpr.MPB); i++ {
		mpr.MPB[i] = -1
	}

	var c, c_cmp float64
	var offsetMax int
	s1 := make([]float64, mp.W)
	s2 := make([]float64, mp.W)
	for diag := idx; diag < idx+batchSize; diag++ {
		if diag >= lenB {
			break
		}

		//for i := 0; i < mp.W; i++ {
		//	c += (mp.B[diag+i] - mub[diag]) * (mp.A[i] - mua[0])
		//}
		copy(s1, mp.B[diag:diag+mp.W])
		copy(s2, mp.A[:mp.W])
		floats.AddConst(-mub[diag], s1)
		floats.AddConst(mua[0], s2)
		c = floats.Dot(s1, s2)

		offsetMax = lenB - diag
		if offsetMax > lenA {
			offsetMax = lenA
		}

		for offset := 0; offset < offsetMax; offset++ {
			c += dfa[offset]*dgb[offset+diag] + dfb[offset+diag]*dga[offset]
			c_cmp = c * (siga[offset] * sigb[offset+diag])
			if mp.Opts.RemapNegCorr && c_cmp < 0 {
				c_cmp = -c_cmp
			}
			if c_cmp > mpr.MP[offset] {
				mpr.MP[offset] = c_cmp
				mpr.Idx[offset] = offset + diag
			}
			if c_cmp > mpr.MPB[offset+diag] {
				mpr.MPB[offset+diag] = c_cmp
				mpr.IdxB[offset+diag] = offset
			}
		}
	}

	if mp.Opts.Euclidean {
		util.P2E(mpr.MP, mp.W)
		util.P2E(mpr.MPB, mp.W)
	}

	return mpr
}

// Analyze performs the matrix profile computation and discovers various features
// from the profile such as motifs, discords, and segmentation. The results are
// visualized and saved into an output file.
func (mp MatrixProfile) Analyze(mo *MPOpts, ao *AnalyzeOpts) error {
	var err error

	if err = mp.Compute(mo); err != nil {
		return err
	}

	if ao == nil {
		ao = NewAnalyzeOpts()
	}

	_, _, cac := mp.DiscoverSegments()

	motifs, err := mp.DiscoverMotifs(ao.KMotifs, ao.RMotifs)
	if err != nil {
		return err
	}

	discords, err := mp.DiscoverDiscords(ao.KDiscords, mp.W/2)
	if err != nil {
		return err
	}

	return mp.Visualize(ao.OutputFilename, motifs, discords, cac)
}

// DiscoverMotifs will iteratively go through the matrix profile to find the
// top k motifs with a given radius. Only applies to self joins.
func (mp MatrixProfile) DiscoverMotifs(k int, r float64) ([]MotifGroup, error) {
	if !mp.SelfJoin {
		return nil, errors.New("can only find top motifs if a self join is performed")
	}
	var err error
	var minDistIdx int

	motifs := make([]MotifGroup, k)

	mpCurrent, _, err := mp.ApplyAV()
	if err != nil {
		return nil, err
	}

	if mp.BF == nil {
		if err = mp.initCaches(); err != nil {
			return nil, err
		}
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
		util.ApplyExclusionZone(prof, initialMotif[0], mp.W/2)
		util.ApplyExclusionZone(prof, initialMotif[1], mp.W/2)
		if j > 0 {
			for k := j; k >= 0; k-- {
				for _, idx := range motifs[k].Idx {
					util.ApplyExclusionZone(prof, idx, mp.W/2)
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
				util.ApplyExclusionZone(prof, minDistIdx, mp.W/2)
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
			util.ApplyExclusionZone(mpCurrent, idx, mp.W/2)
		}

		// sorts the indices in ascending order
		sort.IntSlice(motifs[j].Idx).Sort()
	}

	return motifs[:j], nil
}

// DiscoverDiscords finds the top k time series discords starting indexes from a computed
// matrix profile. Each discovery of a discord will apply an exclusion zone around
// the found index so that new discords can be discovered.
func (mp MatrixProfile) DiscoverDiscords(k int, exclusionZone int) ([]int, error) {
	mpCurrent, _, err := mp.ApplyAV()
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

// DiscoverSegments finds the the index where there may be a potential timeseries
// change. Returns the index of the potential change, value of the corrected
// arc curve score and the histogram of all the crossings for each index in
// the matrix profile index. This approach is based on the UCR paper on
// segmentation of timeseries using matrix profiles which can be found
// https://www.cs.ucr.edu/%7Eeamonn/Segmentation_ICDM.pdf
func (mp MatrixProfile) DiscoverSegments() (int, float64, []float64) {
	histo := arcCurve(mp.Idx)

	for i := 0; i < len(histo); i++ {
		if i == 0 || i == len(histo)-1 {
			histo[i] = math.Min(1.0, float64(len(histo)))
		} else {
			histo[i] = math.Min(1.0, histo[i]/iac(float64(i), len(histo)))
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

// Visualize creates a png of the matrix profile given a matrix profile.
func (mp MatrixProfile) Visualize(fn string, motifs []MotifGroup, discords []int, cac []float64) error {
	sigPts := points(mp.A, len(mp.A))
	mpPts := points(mp.MP, len(mp.A))
	cacPts := points(cac, len(mp.A))
	motifPts := make([][]plotter.XYs, len(motifs))
	discordPts := make([]plotter.XYs, len(discords))
	discordLabels := make([]string, len(discords))

	for i := 0; i < len(motifs); i++ {
		motifPts[i] = make([]plotter.XYs, len(motifs[i].Idx))
	}

	for i := 0; i < len(motifs); i++ {
		for j, idx := range motifs[i].Idx {
			motifPts[i][j] = points(mp.A[idx:idx+mp.W], mp.W)
		}
	}

	for i, idx := range discords {
		discordPts[i] = points(mp.A[idx:idx+mp.W], mp.W)
		discordLabels[i] = strconv.Itoa(idx)
	}

	return plotMP(sigPts, mpPts, cacPts, motifPts, discordPts, discordLabels, fn)
}
