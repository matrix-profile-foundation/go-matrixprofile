package matrixprofile

import (
	"fmt"
	"gonum.org/v1/gonum/stat"
	"math"
	"math/rand"
)

// zNormalize computes a z-normalized version of a slice of floats.
// This is represented by y[i] = x[i] - mean(x)/std(x)
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

// movstd computes the standard deviation of each sliding window of m
// over a slice of floats. This is done by one pass through the data
// and keeping track of the cumulative sum and cumulative sum squared.
// Diffs between these at intervals of m provide a total of O(n)
// calculations for the standard deviation of each window of size m for
// the time series ts.
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

// generateSin produces a sin wave with a given amplitude, frequency,
// phase, sampleRate and duration in seconds
func generateSin(amp, freq, phase, offset, sampleRate, durationSec float64) []float64 {
	nsamp := int(sampleRate * durationSec)
	if nsamp == 0 {
		return nil
	}

	out := make([]float64, nsamp)
	for i := 0; i < nsamp; i++ {
		out[i] = amp*math.Sin(2*math.Pi*freq*float64(i)/sampleRate+phase) + offset
	}
	return out
}

// generateSawtooth produces a sawtooth wave with a given amplitude,
// frequency, phase, sampleRate and duration in seconds
func generateSawtooth(amp, freq, phase, offset, sampleRate, durationSec float64) []float64 {
	nsamp := int(sampleRate * durationSec)
	if nsamp == 0 {
		return nil
	}

	out := make([]float64, nsamp)
	for i := 0; i < nsamp; i++ {
		out[i] = -2*amp/math.Pi*math.Atan(1.0/math.Tan(float64(i)/sampleRate*math.Pi*freq)) + offset
	}
	return out
}

// generateLine creates a line given a slope, offset and number of data points
func generateLine(slope, offset float64, n int) []float64 {
	out := make([]float64, n)
	for i := 0; i < n; i++ {
		out[i] = slope*float64(i) + offset
	}
	return out
}

// generateNoise creates a noise signal
func generateNoise(amp float64, n int) []float64 {
	out := make([]float64, n)
	for i := 0; i < n; i++ {
		out[i] = amp * (rand.Float64() - 0.5)
	}
	return out
}

// sigAdd adds one or more slices of floats together returning a signal
// with a length equal to the longest signal passed in
func sigAdd(sig ...[]float64) []float64 {
	var maxLen int
	for _, signal := range sig {
		if len(signal) > maxLen {
			maxLen = len(signal)
		}
	}
	out := make([]float64, maxLen)
	for _, signal := range sig {
		for i, val := range signal {
			out[i] += val
		}
	}
	return out
}
