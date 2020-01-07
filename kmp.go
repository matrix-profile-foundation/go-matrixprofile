package matrixprofile

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"math"
	"os"
	"sort"

	"github.com/matrix-profile-foundation/go-matrixprofile/util"
	"gonum.org/v1/gonum/fourier"
	"gonum.org/v1/plot/plotter"
)

// KMP is a struct that tracks the current k-dimensional matrix profile
// computation for a given slice of timeseries of length N and subsequence length of M.
// The profile and the profile index are stored here.
type KMP struct {
	T     [][]float64    // a set of timeseries where the number of row represents the number of dimensions and each row is a separate time series
	tMean [][]float64    // sliding mean of each timeseries with a window of m each
	tStd  [][]float64    // sliding standard deviation of each timeseries with a window of m each
	tF    [][]complex128 // holds an existing calculation of the FFT for each timeseries
	n     int            // length of the timeseries
	M     int            // length of a subsequence
	MP    [][]float64    // matrix profile
	Idx   [][]int        // matrix profile index
}

// NewKMP creates a matrix profile struct specifically to be used with the k dimensional
// matrix profile computation. The number of rows represents the number of dimensions,
// and each row holds a series of points of equal length as each other.
func NewKMP(t [][]float64, m int) (*KMP, error) {
	if t == nil || len(t) == 0 {
		return nil, fmt.Errorf("slice is nil or has a length of 0 dimensions")
	}

	k := KMP{
		T: t,
		M: m,
		n: len(t[0]),
	}

	// checks that all timeseries have the same length
	for d := 0; d < len(t); d++ {
		if len(t[d]) != k.n {
			return nil, fmt.Errorf("timeseries %d has a length of %d and doesn't match the first timeseries with length %d", d, len(t[d]), k.n)
		}
	}

	if k.M*2 >= k.n {
		return nil, fmt.Errorf("subsequence length must be less than half the timeseries")
	}

	if k.M < 2 {
		return nil, fmt.Errorf("subsequence length must be at least 2")
	}

	k.tMean = make([][]float64, len(t))
	k.tStd = make([][]float64, len(t))
	k.tF = make([][]complex128, len(t))
	k.MP = make([][]float64, len(t))
	k.Idx = make([][]int, len(t))
	for d := 0; d < len(t); d++ {
		k.tMean[d] = make([]float64, k.n-k.M+1)
		k.tStd[d] = make([]float64, k.n-k.M+1)
		k.tF[d] = make([]complex128, k.n-k.M+1)
		k.MP[d] = make([]float64, k.n-k.M+1)
		k.Idx[d] = make([]int, k.n-k.M+1)
	}

	for d := 0; d < len(t); d++ {
		for i := 0; i < k.n-k.M+1; i++ {
			k.MP[d][i] = math.Inf(1)
			k.Idx[d][i] = math.MaxInt64
		}
	}

	if err := k.initCaches(); err != nil {
		return nil, err
	}

	return &k, nil
}

// Save will save the current matrix profile struct to disk
func (k KMP) Save(filepath, format string) error {
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
		out, err := json.Marshal(k)
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
func (k *KMP) Load(filepath, format string) error {
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
		err = json.Unmarshal(b, k)
	default:
		return fmt.Errorf("invalid load format, %s", format)
	}
	return err
}

// initCaches initializes cached data including the timeseries a and b rolling mean
// and standard deviation and full fourier transform of timeseries b
func (k *KMP) initCaches() error {
	var err error
	// precompute the mean and standard deviation for each window of size m for all
	// sliding windows across the b timeseries
	for d := 0; d < len(k.T); d++ {
		k.tMean[d], k.tStd[d], err = util.MovMeanStd(k.T[d], k.M)
		if err != nil {
			return err
		}
	}

	// precompute the fourier transform of the b timeseries since it will
	// be used multiple times while computing the matrix profile
	fft := fourier.NewFFT(k.n)
	for d := 0; d < len(k.T); d++ {
		k.tF[d] = fft.Coefficients(nil, k.T[d])
	}

	return nil
}

// Compute runs a k dimensional matrix profile calculation across all time series
func (k *KMP) Compute() error {
	return k.mStomp()
}

// MStomp computes the k dimensional matrix profile
func (k *KMP) mStomp() error {
	var err error

	// save the first dot product of the first row that will be used by all future
	// go routines
	cachedDots := make([][]float64, len(k.T))
	fft := fourier.NewFFT(k.n)
	k.crossCorrelate(0, fft, cachedDots)

	var D [][]float64
	D = make([][]float64, len(k.T))
	for d := 0; d < len(D); d++ {
		D[d] = make([]float64, k.n-k.M+1)
	}

	dots := make([][]float64, len(k.T))
	for d := 0; d < len(dots); d++ {
		dots[d] = make([]float64, k.n-k.M+1)
		copy(dots[d], cachedDots[d])
	}

	for idx := 0; idx < k.n-k.M+1; idx++ {
		for d := 0; d < len(dots); d++ {
			if idx > 0 {
				for j := k.n - k.M; j > 0; j-- {
					dots[d][j] = dots[d][j-1] - k.T[d][j-1]*k.T[d][idx-1] + k.T[d][j+k.M-1]*k.T[d][idx+k.M-1]
				}
				dots[d][0] = cachedDots[d][idx]
			}

			for i := 0; i < k.n-k.M+1; i++ {
				D[d][i] = math.Sqrt(2 * float64(k.M) * math.Abs(1-(dots[d][i]-float64(k.M)*k.tMean[d][i]*k.tMean[d][idx])/(float64(k.M)*k.tStd[d][i]*k.tStd[d][idx])))
			}
			// sets the distance in the exclusion zone to +Inf
			util.ApplyExclusionZone(D[d], idx, k.M/2)
		}

		k.columnWiseSort(D)
		k.columnWiseCumSum(D)

		for d := 0; d < len(D); d++ {
			for i := 0; i < k.n-k.M+1; i++ {
				if D[d][i]/(float64(d)+1) < k.MP[d][i] {
					k.MP[d][i] = D[d][i] / (float64(d) + 1)
					k.Idx[d][i] = idx
				}
			}
		}
	}

	return err
}

// crossCorrelate computes the sliding dot product between two slices
// given a query and time series. Uses fast fourier transforms to compute
// the necessary values. Returns the a slice of floats for the cross-correlation
// of the signal q and the k.b signal. This makes an optimization where the query
// length must be less than half the length of the timeseries, b.
func (k KMP) crossCorrelate(idx int, fft *fourier.FFT, D [][]float64) {
	qpad := make([]float64, k.n)
	var qf []complex128
	var dot []float64

	for d := 0; d < len(D); d++ {
		for i := 0; i < k.M; i++ {
			qpad[i] = k.T[d][idx+k.M-i-1]
		}
		qf = fft.Coefficients(nil, qpad)

		// in place multiply the fourier transform of the b time series with
		// the subsequence fourier transform and store in the subsequence fft slice
		for i := 0; i < len(qf); i++ {
			qf[i] = k.tF[d][i] * qf[i]
		}

		dot = fft.Sequence(nil, qf)

		for i := 0; i < k.n-k.M+1; i++ {
			dot[k.M-1+i] = dot[k.M-1+i] / float64(k.n)
		}
		D[d] = dot[k.M-1:]
	}
}

func (k KMP) columnWiseSort(D [][]float64) {
	dist := make([]float64, len(D))
	for i := 0; i < k.n-k.M+1; i++ {
		for d := 0; d < len(D); d++ {
			dist[d] = D[d][i]
		}
		sort.Float64s(dist)
		for d := 0; d < len(D); d++ {
			D[d][i] = dist[d]
		}
	}
}

func (k KMP) columnWiseCumSum(D [][]float64) {
	for d := 0; d < len(D); d++ {
		// change D to be a cumulative sum of distances across dimensions
		if d > 0 {
			for i := 0; i < k.n-k.M+1; i++ {
				D[d][i] += D[d-1][i]
			}
		}
	}
}

// Analyze has not been implemented yet
func (k KMP) Analyze(mo *MPOptions, ao *AnalyzeOptions) error {
	return errors.New("Analyze for KMP has not been implemented yet.")
}

// DiscoverMotifs has not been implemented yet
func (k KMP) DiscoverMotifs(kMotifs int, r float64) ([]MotifGroup, error) {
	return nil, errors.New("Motifs for KMP has not been implemented yet.")
}

// DiscoverDiscords has not been implemented yet
func (k KMP) DiscoverDiscords(kDiscords int, exclusionZone int) ([]int, error) {
	return nil, errors.New("Discords for KMP has not been implemented yet.")
}

// DiscoverSegments has not been implemented yet
func (k KMP) DiscoverSegments() (int, float64, []float64) {
	return 0, 0, nil
}

// Visualize creates a png of the k-dimensional matrix profile.
func (k KMP) Visualize(fn string) error {
	sigPts := make([]plotter.XYs, len(k.T))
	for i := 0; i < len(k.T); i++ {
		sigPts[i] = points(k.T[i], len(k.T[0]))
	}

	mpPts := make([]plotter.XYs, len(k.MP))
	for i := 0; i < len(k.MP); i++ {
		mpPts[i] = points(k.MP[i], len(k.T[0]))
	}

	return plotKMP(sigPts, mpPts, fn)
}
