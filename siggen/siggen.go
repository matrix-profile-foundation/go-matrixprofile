// Package siggen provides basic timeseries generation wrappers
package siggen

import (
	"math"
	"math/rand"
)

// Sin produces a sin wave with a given amplitude, frequency,
// phase, sampleRate and duration in seconds
func Sin(amp, freq, phase, offset, sampleRate, durationSec float64) []float64 {
	nsamp := int(sampleRate * durationSec)
	out := make([]float64, nsamp)
	for i := 0; i < nsamp; i++ {
		out[i] = amp*math.Sin(2*math.Pi*freq*float64(i)/sampleRate+phase) + offset
	}
	return out
}

// Sawtooth produces a sawtooth wave with a given amplitude,
// frequency, phase, sampleRate and duration in seconds
func Sawtooth(amp, freq, phase, offset, sampleRate, durationSec float64) []float64 {
	nsamp := int(sampleRate * durationSec)
	out := make([]float64, nsamp)
	for i := 0; i < nsamp; i++ {
		out[i] = -2*amp/math.Pi*math.Atan(1.0/math.Tan(float64(i)/sampleRate*math.Pi*freq)) + offset
	}
	return out
}

// Square produces a square wave with a given amplitude,
// frequency, phase, sampleRate and duration in seconds
func Square(amp, freq, phase, offset, sampleRate, durationSec float64) []float64 {
	nsamp := int(sampleRate * durationSec)
	out := make([]float64, nsamp)
	var val float64
	for i := 0; i < nsamp; i++ {
		val = math.Sin(2*math.Pi*freq*float64(i)/sampleRate + phase)
		switch {
		case val > 0:
			out[i] = amp + offset
		case val < 0:
			out[i] = -amp + offset
		default:
			out[i] = offset
		}
	}
	return out
}

// Line creates a line given a slope, offset and number of data points
func Line(slope, offset float64, n int) []float64 {
	out := make([]float64, n)
	for i := 0; i < n; i++ {
		out[i] = slope*float64(i) + offset
	}
	return out
}

// Noise creates a noise signal centered around 0
func Noise(amp float64, n int) []float64 {
	out := make([]float64, n)
	for i := 0; i < n; i++ {
		out[i] = amp * (rand.Float64() - 0.5)
	}
	return out
}

// Add adds one or more slices of floats together returning a signal
// with a length equal to the longest signal passed in
func Add(sig ...[]float64) []float64 {
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

// Append adds a series of signals to the input signal extending the length
func Append(in []float64, sig ...[]float64) []float64 {
	totalLen := len(in)
	for _, signal := range sig {
		totalLen += len(signal)
	}

	out := make([]float64, totalLen)
	currIdx := 0
	copy(out[:len(in)], in)
	currIdx += len(in)
	for _, signal := range sig {
		copy(out[currIdx:currIdx+len(signal)], signal)
		currIdx += len(signal)
	}
	return out
}
