package matrixprofile

import (
	"math"
	"math/rand"
)

// generateSin produces a sin wave with a given amplitude, frequency, phase, sampleRate and duration in seconds
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

// generateSawtooth produces a sawtooth wave with a given amplitude, frequency, phase, sampleRate and duration in seconds
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

// sigAdd adds one or more slices of floats together returning a signal with a length equal to the longest signal passed in
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
