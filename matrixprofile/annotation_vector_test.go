package matrixprofile

import (
	"math"
	"testing"
)

func TestMakeDefaultAV(t *testing.T) {
	testdata := []struct {
		d        []float64
		m        int
		expected []float64
	}{
		{[]float64{0, 1, 2, 3, 4, 5}, 3, []float64{1, 1, 1, 1}},
	}
	for _, d := range testdata {
		out := MakeDefaultAV(d.d, d.m)

		if len(out) != len(d.expected) {
			t.Errorf("Expected length %d, but got %d for %v", len(d.expected), len(out), d)
			break
		}

		for i, val := range out {
			if math.Abs(val-d.expected[i]) > 1e-7 {
				t.Errorf("Expected value of %.3f, but got %.3f for %v", d.expected[i], val, d)
			}
		}
	}
}

func TestMakeCompexityAV(t *testing.T) {
	testdata := []struct {
		d        []float64
		m        int
		expected []float64
	}{
		{[]float64{3, 3, 3, 3, 3, 3}, 3, []float64{0, 0, 0, 0}},
		{[]float64{0, 1, 2, 3, 4, 5}, 3, []float64{0, 0, 0, 0}},
		{[]float64{0, 3, 0, 2, 0, 1}, 3, []float64{0.47295372330527, 0.32279030890406757, 0.13962038997193682, 0}},
	}
	for _, d := range testdata {
		out := MakeCompexityAV(d.d, d.m)

		if len(out) != len(d.expected) {
			t.Errorf("Expected length %d, but got %d for %v", len(d.expected), len(out), d)
			break
		}

		for i, val := range out {
			if math.Abs(val-d.expected[i]) > 1e-7 {
				t.Errorf("Expected value of %.3f, but got %.3f for %v", d.expected[i], val, d)
			}
		}
	}
}

func TestMakeMeanStdAV(t *testing.T) {
	testdata := []struct {
		d        []float64
		m        int
		expected []float64
	}{
		{[]float64{3, 3, 3, 3, 3, 3}, 3, []float64{0, 0, 0, 0}},
		{[]float64{-10, 10, -10, 1, -1, 1}, 3, []float64{0, 0, 1, 1}},
		{[]float64{0, 3, 0, 2, 0, 1}, 3, []float64{0, 0, 1, 1}},
	}
	for _, d := range testdata {
		out := MakeMeanStdAV(d.d, d.m)

		if len(out) != len(d.expected) {
			t.Errorf("Expected length %d, but got %d for %v", len(d.expected), len(out), d)
			break
		}

		for i, val := range out {
			if math.Abs(val-d.expected[i]) > 1e-7 {
				t.Errorf("Expected value of %.3f, but got %.3f for %v", d.expected[i], val, d)
			}
		}
	}
}

func TestMakeClippingAV(t *testing.T) {
	testdata := []struct {
		d        []float64
		m        int
		expected []float64
	}{
		{[]float64{3, 3, 3, 3, 3, 3}, 3, []float64{0, 0, 0, 0}},
		{[]float64{0, 1, 2, 3, 4, 5}, 3, []float64{0, 1, 1, 0}},
		{[]float64{0, 3, 0, 2, 0, 1}, 3, []float64{0, 0.5, 0.5, 1}},
	}
	for _, d := range testdata {
		out := MakeClippingAV(d.d, d.m)

		if len(out) != len(d.expected) {
			t.Errorf("Expected length %d, but got %d for %v", len(d.expected), len(out), d)
			break
		}

		for i, val := range out {
			if math.Abs(val-d.expected[i]) > 1e-7 {
				t.Errorf("Expected value of %.3f, but got %.3f for %v", d.expected[i], val, d)
			}
		}
	}
}
