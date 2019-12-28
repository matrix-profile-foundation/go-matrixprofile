package matrixprofile

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"sync"

	"github.com/matrix-profile-foundation/go-matrixprofile/util"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/fourier"
)

type Algo string

const (
	AlgoSTOMP Algo = "STOMP"
	AlgoSTAMP Algo = "STAMP"
	AlgoSTMP  Algo = "STMP"
	AlgoMPX   Algo = "MPX"
	AlgoPMP   Algo = "PMP"
)

// ComputeOptions are parameters to vary the algorithm to compute the matrix profile.
type ComputeOptions struct {
	Algorithm   Algo    // choose which algorithm to compute the matrix profile
	Sample      float64 // only applicable to algorithm STAMP
	Parallelism int
	LowerM      int // used for pan matrix profile
	UpperM      int // used for pan matrix profile
}

// NewComputeOpts returns a default ComputeOptions defaulting to the STOMP algorithm with
// a parallelism of 4.
func NewComputeOpts() ComputeOptions {
	return ComputeOptions{
		Algorithm:   AlgoMPX,
		Sample:      1.0,
		Parallelism: runtime.NumCPU(),
	}
}

// Compute calculate the matrixprofile given a set of input options. This defaults to using
// STOMP unless specified differently
func (mp *MatrixProfile) Compute(o ComputeOptions) error {
	switch o.Algorithm {
	case AlgoSTOMP:
		return mp.stomp(o.Parallelism)
	case AlgoSTAMP:
		return mp.stamp(o.Sample, o.Parallelism)
	case AlgoSTMP:
		return mp.stmp()
	case AlgoMPX:
		return mp.mpx(o.Parallelism)
	case AlgoPMP:
		return mp.pmp(o.LowerM, o.UpperM, o.Sample, o.Parallelism)
	}
	return nil
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

// mpResult is the output struct from a batch processing for STAMP, STOMP, and MPX. This struct
// can later be merged together in linear time or with a divide and conquer approach
type mpResult struct {
	MP  []float64
	Idx []int
	Err error
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

func (mp *MatrixProfile) mpx(parallelism int) error {
	lenA := len(mp.A) - mp.M + 1
	lenB := len(mp.B) - mp.M + 1

	mp.MP = make([]float64, lenA)
	mp.Idx = make([]int, lenA)
	for i := 0; i < len(mp.MP); i++ {
		mp.MP[i] = math.Inf(1)
		mp.Idx[i] = math.MaxInt64
	}

	mua, siga := util.MuInvN(mp.A, mp.M)
	mub, sigb := mua, siga
	if !mp.SelfJoin {
		mub, sigb = util.MuInvN(mp.B, mp.M)
	}

	dfa := make([]float64, lenA)
	dga := make([]float64, lenA)
	for i := 0; i < lenA-1; i++ {
		dfa[i+1] = 0.5 * (mp.A[mp.M+i] - mp.A[i])
		dga[i+1] = (mp.A[mp.M+i] - mua[1+i]) + (mp.A[i] - mua[i])
	}

	dfb, dgb := dfa, dga
	if !mp.SelfJoin {
		dfb = make([]float64, lenB)
		dgb = make([]float64, lenB)
		for i := 0; i < lenB-1; i++ {
			dfb[i+1] = 0.5 * (mp.B[mp.M+i] - mp.B[i])
			dgb[i+1] = (mp.B[mp.M+i] - mub[1+i]) + (mp.B[i] - mub[i])
		}
	}

	// setup for AB join
	batchSize := lenA/parallelism + 1
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
			if mp.SelfJoin {
				results[idx] <- mp.mpxBatch(idx, mua, siga, dfa, dga, batchSize, &wg)
			} else {
				results[idx] <- mp.mpxabBatch(idx, mua, siga, dfa, dga, mub, sigb, dfb, dgb, batchSize, &wg)
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
	batchSize = lenB/parallelism + 1
	results = make([]chan *mpResult, parallelism)
	for i := 0; i < parallelism; i++ {
		results[i] = make(chan *mpResult)
	}

	// go routine to continually check for results on the slice of channels
	// for each batch kicked off. This merges the results of the batched go
	// routines by picking the lowest value in each batch's matrix profile and
	// updating the matrix profile index.
	go func() {
		err = mp.mergeMPResults(results)
		done <- true
	}()

	// kick off multiple go routines to process a batch of rows returning back
	// the matrix profile for that batch and any error encountered
	wg.Add(parallelism)
	for batch := 0; batch < parallelism; batch++ {
		go func(idx int) {
			results[idx] <- mp.mpxbaBatch(idx, mua, siga, dfa, dga, mub, sigb, dfb, dgb, batchSize, &wg)
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
	if idx*batchSize+mp.M/4 > len(mp.A)-mp.M+1 {
		// got an index larger than max lag so ignore
		return &mpResult{}
	}

	mpr := &mpResult{
		MP:  make([]float64, len(mp.A)-mp.M+1),
		Idx: make([]int, len(mp.A)-mp.M+1),
	}
	for i := 0; i < len(mpr.MP); i++ {
		mpr.MP[i] = -1
	}

	var c, c_cmp float64
	s1 := make([]float64, mp.M)
	s2 := make([]float64, mp.M)
	for diag := idx*batchSize + mp.M/4; diag < (idx+1)*batchSize+mp.M/4; diag++ {
		if diag >= len(mp.A)-mp.M+1 {
			break
		}

		//for i := 0; i < mp.M; i++ {
		//	c += (mp.A[diag+i] - mu[diag]) * (mp.A[i] - mu[0])
		//}
		copy(s1, mp.A[diag:diag+mp.M])
		copy(s2, mp.A[:mp.M])
		floats.AddConst(-mu[diag], s1)
		floats.AddConst(mu[0], s2)
		c = floats.Dot(s1, s2)

		for offset := 0; offset < len(mp.A)-mp.M-diag+1; offset++ {
			c += df[offset]*dg[offset+diag] + df[offset+diag]*dg[offset]
			c_cmp = c * (sig[offset] * sig[offset+diag])
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

	for i := 0; i < len(mpr.MP); i++ {
		if mpr.MP[i] > 1 {
			mpr.MP[i] = 1
		}
		mpr.MP[i] = math.Sqrt(2 * float64(mp.M) * (1 - mpr.MP[i]))
	}

	return mpr
}

// mpxabBatch processes a batch set of rows in matrix profile AB join calculation.
func (mp MatrixProfile) mpxabBatch(idx int, mua, siga, dfa, dga, mub, sigb, dfb, dgb []float64, batchSize int, wg *sync.WaitGroup) *mpResult {
	defer wg.Done()
	lenA := len(mp.A) - mp.M + 1
	lenB := len(mp.B) - mp.M + 1

	if idx*batchSize > lenA {
		// got an index larger than max lag so ignore
		return &mpResult{}
	}

	mpr := &mpResult{
		MP:  make([]float64, lenA),
		Idx: make([]int, lenA),
	}
	for i := 0; i < len(mpr.MP); i++ {
		mpr.MP[i] = -1
	}

	var c, c_cmp float64
	var offsetMax int
	s1 := make([]float64, mp.M)
	s2 := make([]float64, mp.M)
	for diag := idx * batchSize; diag < (idx+1)*batchSize; diag++ {
		if diag >= lenA {
			break
		}

		//for i := 0; i < mp.M; i++ {
		//	c += (mp.A[diag+i] - mua[diag]) * (mp.B[i] - mub[0])
		//}
		copy(s1, mp.A[diag:diag+mp.M])
		copy(s2, mp.B[:mp.M])
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
			if c_cmp > mpr.MP[offset+diag] {
				mpr.MP[offset+diag] = c_cmp
				mpr.Idx[offset+diag] = offset
			}
		}
	}

	for i := 0; i < len(mpr.MP); i++ {
		if mpr.MP[i] > 1 {
			mpr.MP[i] = 1
		}
		mpr.MP[i] = math.Sqrt(2 * float64(mp.M) * (1 - mpr.MP[i]))
	}

	return mpr
}

// mpxbaBatch processes a batch set of rows in matrix profile calculation.
func (mp MatrixProfile) mpxbaBatch(idx int, mua, siga, dfa, dga, mub, sigb, dfb, dgb []float64, batchSize int, wg *sync.WaitGroup) *mpResult {
	defer wg.Done()
	lenA := len(mp.A) - mp.M + 1
	lenB := len(mp.B) - mp.M + 1

	if idx*batchSize > lenA {
		// got an index larger than max lag so ignore
		return &mpResult{}
	}

	mpr := &mpResult{
		MP:  make([]float64, lenA),
		Idx: make([]int, lenA),
	}
	for i := 0; i < len(mpr.MP); i++ {
		mpr.MP[i] = -1
	}

	var c, c_cmp float64
	var offsetMax int
	s1 := make([]float64, mp.M)
	s2 := make([]float64, mp.M)
	for diag := idx * batchSize; diag < (idx+1)*batchSize; diag++ {
		if diag >= lenB {
			break
		}

		//for i := 0; i < mp.M; i++ {
		//	c += (mp.B[diag+i] - mub[diag]) * (mp.A[i] - mua[0])
		//}
		copy(s1, mp.B[diag:diag+mp.M])
		copy(s2, mp.A[:mp.M])
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
			if c_cmp > mpr.MP[offset] {
				mpr.MP[offset] = c_cmp
				mpr.Idx[offset] = offset + diag
			}
		}
	}

	for i := 0; i < len(mpr.MP); i++ {
		if mpr.MP[i] > 1 {
			mpr.MP[i] = 1
		}
		mpr.MP[i] = math.Sqrt(2 * float64(mp.M) * (1 - mpr.MP[i]))
	}

	return mpr
}

func (mp *MatrixProfile) pmp(lb, ub int, sample float64, parallelism int) error {
	lenA := len(mp.A) - mp.M + 1
	windows := util.BinarySplit(lb, ub)
	windows = windows[:int(float64(len(windows))*sample)]
	mp.PWindows = windows

	mp.PMP = make([][]float64, len(windows))
	mp.PIdx = make([][]int, len(windows))
	for i := 0; i < len(windows); i++ {
		mp.PMP[i] = make([]float64, lenA)
		mp.PIdx[i] = make([]int, lenA)
		for j := 0; j < lenA; j++ {
			mp.PMP[i][j] = math.Inf(1)
			mp.PIdx[i][j] = math.MaxInt64
		}
	}

	for i, m := range windows {
		mp.M = m
		if err := mp.mpx(parallelism); err != nil {
			return err
		}
		copy(mp.PMP[i], mp.MP)
		copy(mp.PIdx[i], mp.Idx)
	}

	return nil
}
