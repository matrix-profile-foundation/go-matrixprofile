package matrixprofile

import (
	"fmt"
	"gonum.org/v1/gonum/fourier"
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

	for i := 0; i < len(mp.MP); i++ {
		mp.MP[i] = math.Inf(1)
		mp.Idx[i] = math.MaxInt64
	}

	var err error
	// precompute the standard deviation for each window of size m for all sliding windows across the b timeseries
	mp.bStd, err = movstd(mp.b, mp.m)

	// precompute the fourier transform of the b timeseries since it will be used multiple times while computing the matrix profile
	mp.fft = fourier.NewFFT(mp.n)
	mp.bF = mp.fft.Coefficients(nil, mp.b)

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
		applyExclusionZone(profile, idx, mp.m/2)
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
