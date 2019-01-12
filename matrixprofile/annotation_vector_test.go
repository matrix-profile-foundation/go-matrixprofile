package matrixprofile

import (
	"math"
	"testing"
)

func TestMakeCompEstAV(t *testing.T) {
	testdata := []struct {
		d        []float64
		m        int
		expected []float64
	}{
		{[]float64{3, 3, 3, 3, 3, 3}, 3, []float64{0, 0, 0, 0}},
		{[]float64{0, 1, 2, 3, 4, 5}, 3, []float64{0, 0, 0, 0}},
		{[]float64{0, 3, 0, 2, 0, 1}, 3, []float64{0.47295372330527, 0.32279030890406757, 0.13962038997193682, 0}},
	}
	//sqrt18 sqrt13 sqrt8 sqrt5
	//sqrt18-sqrt5/sqrt18
	for _, d := range testdata {
		out := MakeCompEstAV(d.d, d.m)

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
