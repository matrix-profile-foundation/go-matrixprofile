// Package matrixprofile computes the matrix profile and matrix profile index of a time series
package matrixprofile

import (
	"fmt"

	"github.com/matrix-profile-foundation/go-matrixprofile/av"
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
	PMP      [][]float64  // pan matrix profile
	PIdx     [][]int      // pan matrix profile index
	PWindows []int        // pan matrix windows used and is aligned with PMP and PIdx
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

	if mp.M > len(mp.A) || mp.M > len(mp.B) {
		return nil, fmt.Errorf("subsequence length must be less than the timeseries")
	}

	if mp.M < 2 {
		return nil, fmt.Errorf("subsequence length must be at least 2")
	}

	mp.AV = av.Default

	return &mp, nil
}

// ApplyAV applies an annotation vector to the current matrix profile. Annotation vector
// values must be between 0 and 1.
func (mp MatrixProfile) ApplyAV() ([]float64, error) {
	avec, err := av.Create(mp.AV, mp.B, mp.M)
	if err != nil {
		return nil, err
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
