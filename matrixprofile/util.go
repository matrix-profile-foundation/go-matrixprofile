package matrixprofile

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/stat"
)

// zNormalize computes a z-normalized version of a slice of floats.
// This is represented by y[i] = (x[i] - mean(x))/std(x)
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

// movmeanstd computes the mean and standard deviation of each sliding
// window of m over a slice of floats. This is done by one pass through
// the data and keeping track of the cumulative sum and cumulative sum
// squared.  s between these at intervals of m provide a total of O(n)
// calculations for the standard deviation of each window of size m for
// the time series ts.
func movmeanstd(ts []float64, m int) ([]float64, []float64, error) {
	if m <= 1 {
		return nil, nil, fmt.Errorf("length of slice must be greater than 1")
	}

	if m > len(ts) {
		return nil, nil, fmt.Errorf("m cannot be greater than length of slice")
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

	mean := make([]float64, len(ts)-m+1)
	std := make([]float64, len(ts)-m+1)
	for i = 0; i < len(ts)-m+1; i++ {
		mean[i] = (c[i+m] - c[i]) / float64(m)
		std[i] = math.Sqrt((csqr[i+m]-csqr[i])/float64(m) - mean[i]*mean[i])
	}

	return mean, std, nil
}

// applyExclusionZone performs an in place operation on a given matrix
// profile setting distances around an index to +Inf
func applyExclusionZone(profile []float64, idx, zoneSize int) {
	startIdx := 0
	if idx-zoneSize > startIdx {
		startIdx = idx - zoneSize
	}
	endIdx := len(profile)
	if idx+zoneSize < endIdx {
		endIdx = idx + zoneSize
	}
	for i := startIdx; i < endIdx; i++ {
		profile[i] = math.Inf(1)
	}
}

// arcCurve computes the arc curve (histogram) which is uncorrected for.
// This loops through the matrix profile index and increments the
// counter for each index that the destination index passes through
// start from the index in the matrix profile index.
func arcCurve(mpIdx []int) []float64 {
	histo := make([]float64, len(mpIdx))
	for i, idx := range mpIdx {
		switch {
		case idx >= len(mpIdx):
		case idx < 0:
			continue
		case idx > i+1:
			for j := i + 1; j < idx; j++ {
				histo[j]++
			}
		case idx < i-1:
			for j := i - 1; j > idx; j-- {
				histo[j]++
			}
		}
	}
	return histo
}

// iac represents the ideal arc curve with a maximum of n/2 and 0 values
// at 0 and n-1. The derived equation to ensure the requirements is
// -(sqrt(2/n)*(x-n/2))^2 + n/2 = y
func iac(x float64, n int) float64 {
	return -math.Pow(math.Sqrt(2/float64(n))*(x-float64(n)/2.0), 2.0) + float64(n)/2.0
}
