package matrixprofile

import (
	"fmt"
	"math"
	//"github.com/mjibson/go-dsp/fft"
	"gonum.org/v1/gonum/stat"
)

// zNormalize computes a z-normalized version of a slice of floats in place
func zNormalize(ts []float64) error {
	var i int

	if len(ts) == 0 {
		return fmt.Errorf("slice does not have any data")
	}

	m := stat.Mean(ts, nil)

	for i = 0; i < len(ts); i++ {
		ts[i] -= m
	}

	m = stat.Mean(ts, nil)
	var std float64
	for _, val := range ts {
		std += (val - float64(m)) * (val - float64(m))
	}
	std = math.Sqrt(std / float64(len(ts)))

	if std == 0 {
		return fmt.Errorf("standard deviation is zero")
	}

	for i = 0; i < len(ts); i++ {
		ts[i] = ts[i] / std
	}

	return nil
}

// movstd computes the standard deviation of each sliding window of m over a slice of floats
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
		out[i] = math.Sqrt((csqr[i+m]-csqr[i])/float64(m) - math.Pow((c[i+m]-c[i])/float64(m), 2.0))
	}

	return out, nil
}
