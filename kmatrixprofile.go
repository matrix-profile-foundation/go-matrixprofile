package matrixprofile

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math"
	"os"
	"sort"

	"github.com/matrix-profile-foundation/go-matrixprofile/util"
	"gonum.org/v1/gonum/fourier"
)

// KMatrixProfile is a struct that tracks the current k-dimensional matrix profile
// computation for a given slice of timeseries of length N and subsequence length of M.
// The profile and the profile index are stored here.
type KMatrixProfile struct {
	T     [][]float64    // a set of timeseries where the number of row represents the number of dimensions and each row is a separate time series
	tMean [][]float64    // sliding mean of each timeseries with a window of m each
	tStd  [][]float64    // sliding standard deviation of each timeseries with a window of m each
	tF    [][]complex128 // holds an existing calculation of the FFT for each timeseries
	n     int            // length of the timeseries
	M     int            // length of a subsequence
	MP    [][]float64    // matrix profile
	Idx   [][]int        // matrix profile index
}

// NewK creates a matrix profile struct specifically to be used with the k dimensional
// matrix profile computation. The number of rows represents the number of dimensions,
// and each row holds a series of points of equal length as each other.
func NewK(t [][]float64, m int) (*KMatrixProfile, error) {
	if t == nil || len(t) == 0 {
		return nil, fmt.Errorf("slice is nil or has a length of 0 dimensions")
	}

	mp := KMatrixProfile{
		T: t,
		M: m,
		n: len(t[0]),
	}

	// checks that all timeseries have the same length
	for d := 0; d < len(t); d++ {
		if len(t[d]) != mp.n {
			return nil, fmt.Errorf("timeseries %d has a length of %d and doesn't match the first timeseries with length %d", d, len(t[d]), mp.n)
		}
	}

	if mp.M*2 >= mp.n {
		return nil, fmt.Errorf("subsequence length must be less than half the timeseries")
	}

	if mp.M < 2 {
		return nil, fmt.Errorf("subsequence length must be at least 2")
	}

	mp.tMean = make([][]float64, len(t))
	mp.tStd = make([][]float64, len(t))
	mp.tF = make([][]complex128, len(t))
	mp.MP = make([][]float64, len(t))
	mp.Idx = make([][]int, len(t))
	for d := 0; d < len(t); d++ {
		mp.tMean[d] = make([]float64, mp.n-mp.M+1)
		mp.tStd[d] = make([]float64, mp.n-mp.M+1)
		mp.tF[d] = make([]complex128, mp.n-mp.M+1)
		mp.MP[d] = make([]float64, mp.n-mp.M+1)
		mp.Idx[d] = make([]int, mp.n-mp.M+1)
	}

	for d := 0; d < len(t); d++ {
		for i := 0; i < mp.n-mp.M+1; i++ {
			mp.MP[d][i] = math.Inf(1)
			mp.Idx[d][i] = math.MaxInt64
		}
	}

	if err := mp.initCaches(); err != nil {
		return nil, err
	}

	return &mp, nil
}

// Save will save the current matrix profile struct to disk
func (mp KMatrixProfile) Save(filepath, format string) error {
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
func (mp *KMatrixProfile) Load(filepath, format string) error {
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

// initCaches initializes cached data including the timeseries a and b rolling mean
// and standard deviation and full fourier transform of timeseries b
func (mp *KMatrixProfile) initCaches() error {
	var err error
	// precompute the mean and standard deviation for each window of size m for all
	// sliding windows across the b timeseries
	for d := 0; d < len(mp.T); d++ {
		mp.tMean[d], mp.tStd[d], err = util.MovMeanStd(mp.T[d], mp.M)
		if err != nil {
			return err
		}
	}

	// precompute the fourier transform of the b timeseries since it will
	// be used multiple times while computing the matrix profile
	fft := fourier.NewFFT(mp.n)
	for d := 0; d < len(mp.T); d++ {
		mp.tF[d] = fft.Coefficients(nil, mp.T[d])
	}

	return nil
}

// Compute runs a k dimensional matrix profile calculation across all time series
func (mp *KMatrixProfile) Compute() error {
	return mp.mStomp()
}

// MStomp computes the k dimensional matrix profile
func (mp *KMatrixProfile) mStomp() error {
	var err error

	// save the first dot product of the first row that will be used by all future
	// go routines
	cachedDots := make([][]float64, len(mp.T))
	fft := fourier.NewFFT(mp.n)
	mp.crossCorrelate(0, fft, cachedDots)

	var D [][]float64
	D = make([][]float64, len(mp.T))
	for d := 0; d < len(D); d++ {
		D[d] = make([]float64, mp.n-mp.M+1)
	}

	dots := make([][]float64, len(mp.T))
	for d := 0; d < len(dots); d++ {
		dots[d] = make([]float64, mp.n-mp.M+1)
		copy(dots[d], cachedDots[d])
	}

	for idx := 0; idx < mp.n-mp.M+1; idx++ {
		for d := 0; d < len(dots); d++ {
			if idx > 0 {
				for j := mp.n - mp.M; j > 0; j-- {
					dots[d][j] = dots[d][j-1] - mp.T[d][j-1]*mp.T[d][idx-1] + mp.T[d][j+mp.M-1]*mp.T[d][idx+mp.M-1]
				}
				dots[d][0] = cachedDots[d][idx]
			}

			for i := 0; i < mp.n-mp.M+1; i++ {
				D[d][i] = math.Sqrt(2 * float64(mp.M) * math.Abs(1-(dots[d][i]-float64(mp.M)*mp.tMean[d][i]*mp.tMean[d][idx])/(float64(mp.M)*mp.tStd[d][i]*mp.tStd[d][idx])))
			}
			// sets the distance in the exclusion zone to +Inf
			util.ApplyExclusionZone(D[d], idx, mp.M/2)
		}

		mp.columnWiseSort(D)
		mp.columnWiseCumSum(D)

		for d := 0; d < len(D); d++ {
			for i := 0; i < mp.n-mp.M+1; i++ {
				if D[d][i]/(float64(d)+1) < mp.MP[d][i] {
					mp.MP[d][i] = D[d][i] / (float64(d) + 1)
					mp.Idx[d][i] = idx
				}
			}
		}
	}

	return err
}

// crossCorrelate computes the sliding dot product between two slices
// given a query and time series. Uses fast fourier transforms to compute
// the necessary values. Returns the a slice of floats for the cross-correlation
// of the signal q and the mp.b signal. This makes an optimization where the query
// length must be less than half the length of the timeseries, b.
func (mp KMatrixProfile) crossCorrelate(idx int, fft *fourier.FFT, D [][]float64) {
	qpad := make([]float64, mp.n)
	var qf []complex128
	var dot []float64

	for d := 0; d < len(D); d++ {
		for i := 0; i < mp.M; i++ {
			qpad[i] = mp.T[d][idx+mp.M-i-1]
		}
		qf = fft.Coefficients(nil, qpad)

		// in place multiply the fourier transform of the b time series with
		// the subsequence fourier transform and store in the subsequence fft slice
		for i := 0; i < len(qf); i++ {
			qf[i] = mp.tF[d][i] * qf[i]
		}

		dot = fft.Sequence(nil, qf)

		for i := 0; i < mp.n-mp.M+1; i++ {
			dot[mp.M-1+i] = dot[mp.M-1+i] / float64(mp.n)
		}
		D[d] = dot[mp.M-1:]
	}
}

func (mp KMatrixProfile) columnWiseSort(D [][]float64) {
	dist := make([]float64, len(D))
	for i := 0; i < mp.n-mp.M+1; i++ {
		for d := 0; d < len(D); d++ {
			dist[d] = D[d][i]
		}
		sort.Float64s(dist)
		for d := 0; d < len(D); d++ {
			D[d][i] = dist[d]
		}
	}
}

func (mp KMatrixProfile) columnWiseCumSum(D [][]float64) {
	for d := 0; d < len(D); d++ {
		// change D to be a cumulative sum of distances across dimensions
		if d > 0 {
			for i := 0; i < mp.n-mp.M+1; i++ {
				D[d][i] += D[d-1][i]
			}
		}
	}
}
