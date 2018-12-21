package matrixprofile

import (
	"math"
	"testing"
)

func TestIac(t *testing.T) {
	testdata := []struct {
		x        float64
		n        int
		expected float64
	}{
		{0, 124, 0},
		{124, 124, 0},
		{62, 124, 62},
	}

	var out float64
	for _, d := range testdata {
		if out = iac(d.x, d.n); out != d.expected {
			t.Errorf("Expected %.3f but got %.3f", d.expected, out)
		}
	}
}

func TestArcCurve(t *testing.T) {
	testdata := []struct {
		mpIdx         []int
		expectedHisto []float64
	}{
		{[]int{}, []float64{}},
		{[]int{1, 1, 1, 1, 1}, []float64{0, 0, 2, 1, 0}},
		{[]int{4, 5, 6, 0, 2, 1, 0}, []float64{0, 3, 5, 6, 4, 2, 0}},
		{[]int{4, 5, 12, 0, 2, 1, 0}, []float64{0, 3, 5, 5, 3, 1, 0}},
		{[]int{4, 5, -1, 0, 2, 1, 0}, []float64{0, 3, 5, 5, 3, 1, 0}},
		{[]int{4, 5, 6, 2, 2, 1, 0}, []float64{0, 2, 4, 6, 4, 2, 0}},
		{[]int{2, 3, 0, 0, 6, 3, 4}, []float64{0, 3, 2, 0, 1, 2, 0}},
	}

	var histo []float64
	for _, d := range testdata {
		histo = arcCurve(d.mpIdx)
		if len(histo) != len(d.expectedHisto) {
			t.Errorf("Expected %d elements, but got %d, %+v", len(d.expectedHisto), len(histo), d)
		}
		for i := 0; i < len(histo); i++ {
			if math.Abs(float64(histo[i]-d.expectedHisto[i])) > 1e-7 {
				t.Errorf("Expected %v,\nbut got\n%v for\n%+v", d.expectedHisto, histo, d)
				break
			}
		}
	}
}

func TestSegment(t *testing.T) {
	testdata := []struct {
		mpIdx         []int
		expectedIdx   int
		expectedVal   float64
		expectedHisto []float64
	}{
		{[]int{}, 0, 0, nil},
		{[]int{1, 1, 1, 1, 1}, 0, 0, nil},
		{[]int{4, 5, 6, 0, 2, 1, 0}, 5, 0.7, []float64{1, 1, 1, 1, 1, 0.7, 1}},
		{[]int{4, 5, 12, 0, 2, 1, 0}, 5, 0.35, []float64{1, 1, 1, 1, 0.875, 0.35, 1}},
		{[]int{4, 5, -1, 0, 2, 1, 0}, 5, 0.35, []float64{1, 1, 1, 1, 0.875, 0.35, 1}},
		{[]int{4, 5, 6, 2, 2, 1, 0}, 5, 0.7, []float64{1, 1, 1, 1, 1, 0.7, 1}},
		{[]int{2, 3, 0, 0, 6, 3, 4}, 3, 0, []float64{1, 1, 0.7, 0, 0.29166666, 0.7, 1}},
	}

	var minIdx int
	var minVal float64
	var histo []float64
	for _, d := range testdata {
		minIdx, minVal, histo = Segment(d.mpIdx)
		if histo != nil && d.expectedHisto == nil {
			// Failed to compute histogram
			continue
		}
		if minIdx != d.expectedIdx {
			t.Errorf("Expected %d min index but got %d, %+v", d.expectedIdx, minIdx, d)
		}
		if minVal != d.expectedVal {
			t.Errorf("Expected %.3f min index value but got %.3f, %+v", d.expectedVal, minVal, d)
		}
		if len(histo) != len(d.expectedHisto) {
			t.Errorf("Expected %d elements, but got %d, %+v", len(d.expectedHisto), len(histo), d)
		}
		for i := 0; i < len(histo); i++ {
			if math.Abs(float64(histo[i]-d.expectedHisto[i])) > 1e-7 {
				t.Errorf("Expected %v,\nbut got\n%v for\n%+v", d.expectedHisto, histo, d)
				break
			}
		}
	}
}
