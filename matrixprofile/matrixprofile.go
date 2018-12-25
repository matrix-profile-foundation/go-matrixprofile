package matrixprofile

import (
	"fmt"
	"gonum.org/v1/gonum/fourier"
	"gonum.org/v1/gonum/stat"
	"math"
	"math/rand"
)

// MatrixProfile is a struct that tracks the current matrix profile computation for a given timeseries of length N and subsequence length of M. The profile and the profile index are stored here.
type MatrixProfile struct {
	a        []float64 // query time series
	b        []float64 // timeseries to perform full join with
	bStd     []float64 // sliding standard deviation of b with a window of m each
	n        int       // length of the timeseries
	m        int       // length of a subsequence
	fft      *fourier.FFT
	selfJoin bool         // indicates whether a self join is performed with an exclusion zone
	bF       []complex128 // holds an existing calculation of the FFT of b timeseries
	MP       []float64    // matrix profile
	Idx      []int        // matrix profile index
}

// New creates a matrix profile struct with a given timeseries length n and subsequence length of m. The first slice, a, is used as the initial timeseries to join with the second, b. If b is nil, then the matrix profile assumes a self join on the first timeseries.
func New(a, b []float64, m int) (*MatrixProfile, error) {
	if a == nil || len(a) == 0 {
		return nil, fmt.Errorf("first slice is nil or has a length of 0")
	}

	if b != nil && len(b) == 0 {
		return nil, fmt.Errorf("second slice must be nil for self-join operation or have a length greater than 0")
	}

	mp := MatrixProfile{
		a: a,
		m: m,
		n: len(b),
	}
	if b == nil {
		mp.MP = make([]float64, len(a)-m+1)
		mp.Idx = make([]int, len(a)-m+1)
		mp.n = len(a)
		mp.b = a
		mp.selfJoin = true
	} else {
		mp.MP = make([]float64, len(b)-m+1)
		mp.Idx = make([]int, len(b)-m+1)
		mp.b = b
	}
	mp.fft = fourier.NewFFT(mp.n)

	for i := 0; i < len(mp.MP); i++ {
		mp.MP[i] = math.Inf(1)
		mp.Idx[i] = math.MaxInt64
	}

	var err error
	mp.bStd, err = movstd(mp.b, mp.m)

	return &mp, err
}

// slidingDotProduct computes the sliding dot product between two slices given a query and time series. Uses fast fourier transforms to compute the necessary values
func (mp *MatrixProfile) slidingDotProduct(q []float64) ([]float64, error) {
	if mp.m*2 >= mp.n {
		return nil, fmt.Errorf("length of query must be less than half the timeseries")
	}

	if mp.m < 2 {
		return nil, fmt.Errorf("query must be at least length 2")
	}

	qpad := make([]float64, mp.n)
	for i := 0; i < len(q); i++ {
		qpad[i] = q[mp.m-i-1]
	}

	if mp.bF == nil {
		mp.bF = mp.fft.Coefficients(nil, mp.b)
	}
	qf := mp.fft.Coefficients(nil, qpad)

	// in place multiply the fourier transform of the b time series with the subsequence fourier transform and store in the subsequence fft slice
	for i := 0; i < len(qf); i++ {
		qf[i] = mp.bF[i] * qf[i]
	}

	dot := mp.fft.Sequence(nil, qf)

	out := make([]float64, mp.n-mp.m+1)
	for i := 0; i < len(out); i++ {
		out[i] = dot[mp.m-1+i] / float64(mp.n)
	}
	return out, nil
}

// mass calculates the Mueen's algorithm for similarity search (MASS) between a specified query and timeseries.
func (mp MatrixProfile) mass(q []float64) ([]float64, error) {
	if mp.m < 2 {
		return nil, fmt.Errorf("need at least 2 samples for the query")
	}

	if mp.m*2 >= mp.n {
		return nil, fmt.Errorf("query must be less than half of the timeseries")
	}

	qnorm, err := zNormalize(q)
	if err != nil {
		return nil, err
	}

	dot, err := mp.slidingDotProduct(qnorm)
	if err != nil {
		return nil, err
	}

	if len(mp.bStd) != len(dot) {
		return nil, fmt.Errorf("length of rolling standard deviation, %d, is not the same as the sliding dot product, %d", len(mp.bStd), len(dot))
	}

	// converting cross correlation value to euclidian distance
	out := make([]float64, len(dot))
	for i := 0; i < len(dot); i++ {
		out[i] = math.Sqrt(math.Abs(2 * (float64(mp.m) - (dot[i] / mp.bStd[i]))))
	}
	return out, nil
}

// distanceProfile computes the distance profile between a and b time series. If b is set to nil then it assumes a self join and will create an exclusion area for trivial nearest neighbors
func (mp MatrixProfile) distanceProfile(idx int) ([]float64, error) {
	if idx+mp.m > len(mp.a) {
		return nil, fmt.Errorf("index %d with m %d asks for data beyond the length of a, %d", idx, mp.m, len(mp.a))
	}

	query := mp.a[idx : idx+mp.m]
	profile, err := mp.mass(query)
	if err != nil {
		return nil, err
	}

	// sets the distance in the exclusion zone to +Inf
	if mp.selfJoin {
		startIdx := 0
		if idx-mp.m/2 > startIdx {
			startIdx = idx - mp.m/2
		}
		endIdx := len(profile)
		if idx+mp.m/2 < endIdx {
			endIdx = idx + mp.m/2
		}
		for i := startIdx; i < endIdx; i++ {
			profile[i] = math.Inf(1)
		}
	}
	return profile, nil
}

// Stmp computes the full matrix profile given two time series as inputs. If the second time series is set to nil then a self join on the first will be performed.
func (mp *MatrixProfile) Stmp() error {
	var err error
	var profile []float64

	for i := 0; i < mp.n-mp.m+1; i++ {
		profile, err = mp.distanceProfile(i)
		if err != nil {
			return err
		}

		if len(profile) != len(mp.MP) {
			return fmt.Errorf("distance profile length, %d, and initialized matrix profile length, %d, do not match", len(profile), len(mp.MP))
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

// Stamp uses random ordering to compute the matrix profile. User can the sample to anything between 0 and 1 so that the computation early terminates and provides the current computed matrix profile. This should compute far faster at the cost of an approximation of the matrix profile
func (mp *MatrixProfile) Stamp(sample float64) error {
	if sample == 0.0 {
		return fmt.Errorf("must provide a non zero sampling")
	}

	var profile []float64
	var i, j int
	var err error

	randIdx := rand.Perm(mp.n - mp.m + 1)
	for i = 0; i < int(float64(mp.n-mp.m+1)*sample); i++ {
		profile, err = mp.distanceProfile(randIdx[i])
		if err != nil {
			return err
		}
		if len(profile) != len(mp.MP) {
			return fmt.Errorf("distance profile length, %d, and initialized matrix profile length, %d, do not match", len(profile), len(mp.MP))
		}
		for j = 0; j < len(profile); j++ {
			if profile[j] <= mp.MP[j] {
				mp.MP[j] = profile[j]
				mp.Idx[j] = randIdx[i]
			}
		}
	}
	return nil
}

// zNormalize computes a z-normalized version of a slice of floats. This is represented by y[i] = x[i] - mean(x)/std(x)
func zNormalize(ts []float64) ([]float64, error) {
	var i int

	if len(ts) == 0 {
		return nil, fmt.Errorf("slice does not have any data")
	}

	m := stat.Mean(ts, nil)

	out := make([]float64, len(ts))
	for i = 0; i < len(ts); i++ {
		out[i] = ts[i] - m
	}

	var std float64
	for _, val := range out {
		std += val * val
	}
	std = math.Sqrt(std / float64(len(out)))

	if std == 0 {
		return out, fmt.Errorf("standard deviation is zero")
	}

	for i = 0; i < len(ts); i++ {
		out[i] = out[i] / std
	}

	return out, nil
}

// movstd computes the standard deviation of each sliding window of m over a slice of floats. This is done by one pass through the data and keeping track of the cumulative sum and cumulative sum squared. Diffs between these at intervals of m provide a total of O(n) calculations for the standard deviation of each window of size m for the time series ts.
func movstd(ts []float64, m int) ([]float64, error) {
	if m <= 1 {
		return nil, fmt.Errorf("length of slice must be greater than 1")
	}

	if m >= len(ts) {
		return nil, fmt.Errorf("m must be less than length of slice")
	}

	var i int

	c := make([]float64, len(ts)+1)
	csqr := make([]float64, len(ts)+1)
	for i = 0; i < len(ts)+1; i++ {
		if i == 0 {
			c[i] = 0
			csqr[i] = 0
		} else {
			c[i] = ts[i-1] + c[i-1]
			csqr[i] = ts[i-1]*ts[i-1] + csqr[i-1]
		}
	}

	out := make([]float64, len(ts)-m+1)
	for i = 0; i < len(ts)-m+1; i++ {
		out[i] = math.Sqrt((csqr[i+m]-csqr[i])/float64(m) - (c[i+m]-c[i])*(c[i+m]-c[i])/float64(m*m))
	}

	return out, nil
}
