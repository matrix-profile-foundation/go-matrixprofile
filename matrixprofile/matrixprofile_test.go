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
			// Got an error and expected an error
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
			// Got an error while calculating and expected an error
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

func TestSlidingDotProducts(t *testing.T) {
	var err error
	var out []float64

	testdata := []struct {
		q        []float64
		t        []float64
		expected []float64
	}{
		{[]float64{}, []float64{}, nil},
		{[]float64{1, 1, 1, 1, 1}, []float64{}, nil},
		{[]float64{}, []float64{1, 1, 1, 1, 1}, nil},
		{[]float64{1, 1}, []float64{1, 1, 1, 1, 1}, []float64{2, 2, 2, 2}},
		{[]float64{1, 2}, []float64{1, 2, 3, 3, 2, 1}, []float64{5, 8, 9, 7, 4}},
		{[]float64{1, 2}, []float64{1, 2, 3, 3, 2, 1, 1}, []float64{5, 8, 9, 7, 4, 3}},
		{[]float64{1, 2, 1}, []float64{1, 2, 3, 4, 3, 2, 1}, []float64{8, 12, 14, 12, 8}},
		{[]float64{1, 2, 1}, []float64{1, 2, 3, 4, 3, 2, 1, 1}, []float64{8, 12, 14, 12, 8, 5}},
	}

	for _, d := range testdata {
		out, err = slidingDotProduct(d.q, d.t)
		if err != nil && d.expected == nil {
			// Got an error while z normalizing and expected an error
			continue
		}
		if d.expected == nil {
			t.Errorf("Expected an invalid sliding dot product calculation, %v", d)
		}
		if err != nil {
			t.Errorf("Did not expect error, %v", err)
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

func TestMass(t *testing.T) {
	var err error
	var out []float64

	testdata := []struct {
		q        []float64
		t        []float64
		expected []float64
	}{
		{[]float64{}, []float64{}, nil},
		{[]float64{1, 1, 1, 1, 1}, []float64{}, nil},
		{[]float64{}, []float64{1, 1, 1, 1, 1}, nil},
		{[]float64{1, 1}, []float64{1, 1, 1, 1, 1}, nil},
		{[]float64{0, 1, 1, 0}, []float64{1e-6, 1e-5, 1e-5, 1e-5, 5, 5, 1e-5, 1e-5, 1e-5, 1e-5, 7, 7, 1e-5, 1e-5},
			[]float64{1.838803373328544, 3.552295335908461, 2.828427124746192, 6.664001874625056e-08, 2.8284271247461885,
				3.5522953359084606, 2.8284271366321914, 3.5522953359084606, 2.82842712474619, 0, 2.82842712474619070}},
	}

	for _, d := range testdata {
		out, err = mass(d.q, d.t)
		if err != nil && d.expected == nil {
			// Got an error while z normalizing and expected an error
			continue
		}
		if d.expected == nil {
			t.Errorf("Expected an invalid mass calculation, %v", d)
		}
		if err != nil {
			t.Errorf("Did not expect error, %v", err)
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
