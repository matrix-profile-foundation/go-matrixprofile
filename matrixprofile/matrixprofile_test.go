package matrixprofile

import (
	"math"
	"testing"
)

func TestZNormalize(t *testing.T) {
	var err error

	testdata := []struct {
		data     []float64
		expected []float64
	}{
		{[]float64{}, nil},
		{[]float64{1, 1, 1, 1}, nil},
		{[]float64{-1, 1, -1, 1}, []float64{-1, 1, -1, 1}},
		{[]float64{7, 5, 5, 7}, []float64{1, -1, -1, 1}},
	}

	for _, d := range testdata {
		err = zNormalize(d.data)
		if err != nil && d.expected == nil {
			// Got an error while z normalizing and expected an error
			continue
		}
		if d.expected == nil {
			t.Errorf("Expected an invalid standard deviation of 0, %v", d)
		}
		if len(d.data) != len(d.expected) {
			t.Errorf("Expected %d elements, but got %d, %v", len(d.expected), len(d.data), d)
		}
		for i := 0; i < len(d.data); i++ {
			if math.Abs(d.data[i]-d.expected[i]) > 1e-14 {
				t.Errorf("Expected %v, but got %v for %v", d.expected, d.data, d)
				break
			}
		}
	}
}

func TestMovstd(t *testing.T) {
	var err error
	var out []float64

	testdata := []struct {
		data     []float64
		m        int
		expected []float64
	}{
		{[]float64{}, 4, nil},
		{[]float64{}, 0, nil},
		{[]float64{1, 1, 1, 1}, 0, nil},
		{[]float64{1, 1, 1, 1}, 2, []float64{0, 0, 0}},
		{[]float64{-1, -1, -1, -1}, 2, []float64{0, 0, 0}},
		{[]float64{1, -1, -1, 1}, 2, []float64{1, 0, 1}},
		{[]float64{1, -1, -1, 1}, 4, nil},
	}

	for _, d := range testdata {
		out, err = movstd(d.data, d.m)
		if err != nil && d.expected == nil {
			// Got an error while z normalizing and expected an error
			continue
		}
		if d.expected == nil {
			t.Errorf("Expected an invalid moving standard deviation, %v", d)
		}
		if len(out) != len(d.expected) {
			t.Errorf("Expected %d elements, but got %d, %v", len(d.expected), len(out), d)
		}
		for i := 0; i < len(out); i++ {
			if math.Abs(out[i]-d.expected[i]) > 1e-14 {
				t.Errorf("Expected %v, but got %v for %v", d.expected, out, d)
				break
			}
		}

	}
}
