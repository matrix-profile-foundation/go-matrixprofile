package matrixprofile

import (
	"fmt"
	"github.com/mjibson/go-dsp/fft"
	"gonum.org/v1/gonum/stat"
	"math"
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

	var std float64
	for _, val := range ts {
		std += val * val
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
		out[i] = math.Sqrt((csqr[i+m]-csqr[i])/float64(m) - (c[i+m]-c[i])*(c[i+m]-c[i])/float64(m*m))
	}

	return out, nil
}

// slidingDotProduct computes the sliding dot product between two slices given a query and time series. Uses fast fourier transforms to compute the necessary values
func slidingDotProduct(q, t []float64) ([]float64, error) {
	m := len(q)
	n := len(t)

	if m*2 >= n {
		return nil, fmt.Errorf("length of query must be less than half the timeseries")
	}

	if m == 0 {
		return nil, fmt.Errorf("query must have a length greater than 0")
	}

	qpad := make([]float64, len(t))
	for i := 0; i < len(q); i++ {
		qpad[i] = q[m-i-1]
	}

	f, err := multComplexSlice(fft.FFTReal(t), fft.FFTReal(qpad))
	if err != nil {
		return nil, err
	}
	dot := fft.IFFT(f)

	out := make([]float64, n-m+1)
	for i := 0; i < len(out); i++ {
		out[i] = float64(real(dot[m-1+i]))
	}
	return out, nil
}

func multComplexSlice(a, b []complex128) ([]complex128, error) {
	if len(a) != len(b) {
		return nil, fmt.Errorf("length of both complex slices are not the same")
	}

	out := make([]complex128, len(a))
	for i := 0; i < len(a); i++ {
		out[i] = a[i] * b[i]
	}
	return out, nil
}

// mass calculates the Mueen's algorithm for similarity search (MASS) between a specified query and timeseries.
func mass(q, t []float64) ([]float64, error) {
	m := len(q)
	n := len(t)

	if m <= 1 {
		return nil, fmt.Errorf("need more than 1 sample for the query")
	}

	if m*2 >= n {
		return nil, fmt.Errorf("query must be less than half of the timeseries")
	}

	if err := zNormalize(q); err != nil {
		return nil, err
	}

	std, err := movstd(t, m)
	if err != nil {
		return nil, err
	}

	dot, err := slidingDotProduct(q, t)
	if err != nil {
		return nil, err
	}

	if len(std) != len(dot) {
		return nil, fmt.Errorf("length of rolling standard deviation, %d, is not the same as the sliding dot product, %d", len(std), len(dot))
	}

	out := make([]float64, len(dot))
	for i := 0; i < len(dot); i++ {
		out[i] = math.Sqrt(2 * (float64(m) - (dot[i] / std[i])))
	}
	return out, nil
}
