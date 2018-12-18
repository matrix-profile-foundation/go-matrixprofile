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
		{[]float64{-1, 1, -1, 1}, []float64{-0.8660254037844387, 0.8660254037844387, -0.8660254037844387, 0.8660254037844387}},
		{[]float64{7, 5, 5, 7}, []float64{0.8660254037844387, -0.8660254037844387, -0.8660254037844387, 0.8660254037844387}},
	}

	for _, d := range testdata {
		err = zNormalize(d.data)
		if err != nil && d.expected == nil {
			// Got an error while z normalizing and expected an error
			continue
		}
		if d.expected == nil {
			t.Errorf("Expected an invalid standard deviation of 0")
		}
		if len(d.data) != len(d.expected) {
			t.Errorf("Expected %d elements, but got %d", len(d.expected), len(d.data))
		}
		for i := 0; i < len(d.data); i++ {
			if math.Abs(d.data[i]-d.expected[i]) > 1e-14 {
				t.Errorf("Expected %.3f, but got %.3f for %v", d.expected[i], d.data[i], d)
				break
			}
		}

	}
}
