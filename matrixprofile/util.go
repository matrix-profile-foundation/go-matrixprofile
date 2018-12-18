package matrixprofile

import (
	"fmt"
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

	std := stat.StdDev(ts, nil)

	if std == 0 {
		return fmt.Errorf("standard deviation is zero")
	}

	for i = 0; i < len(ts); i++ {
		ts[i] = ts[i] / std
	}

	return nil
}
