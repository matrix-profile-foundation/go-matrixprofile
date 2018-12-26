// Package matrixprofile computes the matrix profile and matrix profile index of a time series
package matrixprofile

import (
	"errors"
	"fmt"
	"gonum.org/v1/gonum/fourier"
	"math"
	"math/rand"
	"sort"
)

// MatrixProfile is a struct that tracks the current matrix profile computation
// for a given timeseries of length N and subsequence length of M. The profile
// and the profile index are stored here.
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
	// precompute the standard deviation for each window of size m for all
	// sliding windows across the b timeseries
	mp.bStd, err = movstd(mp.b, mp.m)

	// precompute the fourier transform of the b timeseries since it will
	// be used multiple times while computing the matrix profile
	mp.fft = fourier.NewFFT(mp.n)
	mp.bF = mp.fft.Coefficients(nil, mp.b)

	return &mp, err
}

// slidingDotProduct computes the sliding dot product between two slices
// given a query and time series. Uses fast fourier transforms to compute
// the necessary values. Returns the a slice of floats for the cross-correlation
// of the signal q and the mp.b signal.
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

	// in place multiply the fourier transform of the b time series with
	// the subsequence fourier transform and store in the subsequence fft slice
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

// mass calculates the Mueen's algorithm for similarity search (MASS)
// between a specified query and timeseries. Returns the euclidean distance
// of the query to every subsequence in mp.b as a slice of floats.
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

// distanceProfile computes the distance profile between a and b time series.
// If b is set to nil then it assumes a self join and will create an exclusion
// area for trivial nearest neighbors. Returns the euclidean distance between
// the specified subsequence in mp.a with each subsequence in mp.b as a slice
// of floats
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

// Stmp computes the full matrix profile given two time series as inputs.
// If the second time series is set to nil then a self join on the first
// will be performed. Stores the matrix profile and matrix profile index
// in the struct.
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

// Stamp uses random ordering to compute the matrix profile. User
// can specify the sample to be anything between 0 and 1 so that the
// computation early terminates and provides the current computed matrix
// profile. This should compute far faster at the cost of an approximation
// of the matrix profile. Stores the matrix profile and matrix profile index
// in the struct.
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

// MotifGroup stores a list of indices representing a similar motif along
// with the minimum distance that this set of motif composes of.
type MotifGroup struct {
	Idx     []int
	MinDist float64
}

// TopKMotifs will iteratively go through the matrix profile to find the
// top k motifs with a given radius. Only applies to self joins.
func (mp MatrixProfile) TopKMotifs(k int, r float64) ([]MotifGroup, error) {
	if !mp.selfJoin {
		return nil, errors.New("can only find top motifs if a self join is performed")
	}

	motifs := make([]MotifGroup, k)

	mpCurrent := make([]float64, len(mp.MP))
	copy(mpCurrent, mp.MP)

	for j := 0; j < k; j++ {
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

		for _, idx := range initialMotif {
			prof, err := mp.distanceProfile(idx)
			if err != nil {
				return nil, err
			}
			for i, d := range prof {
				if d < motifDistance*r {
					motifSet[i] = struct{}{}
				}
			}
		}

		// store the found motif indexes and create an exclusion zone around
		// each index in the current matrix profile
		motifs[j] = MotifGroup{
			Idx:     make([]int, 0, len(motifSet)),
			MinDist: motifDistance,
		}
		for idx, _ := range motifSet {
			motifs[j].Idx = append(motifs[j].Idx, idx)
			applyExclusionZone(mpCurrent, idx, mp.m/2)
		}

		// sorts the indices in ascending order
		sort.IntSlice(motifs[j].Idx).Sort()
	}

	return motifs, nil
}

// Segment finds the the index where there may be a potential timeseries
// change. Returns the index of the potential change, value of the corrected
// arc curve score and the histogram of all the crossings for each index in
// the matrix profile index. This approach is based on the UCR paper on
// segmentation of timeseries using matrix profiles which can be found
// https://www.cs.ucr.edu/%7Eeamonn/Segmentation_ICDM.pdf
func (mp MatrixProfile) Segment() (int, float64, []float64) {
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
