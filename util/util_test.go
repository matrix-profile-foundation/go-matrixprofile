package util

import (
	"math"
	"testing"
)

func TestZNormalize(t *testing.T) {
	var out []float64
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
		out, err = ZNormalize(d.data)
		if err != nil && d.expected == nil {
			// Got an error and expected an error
			continue
		}
		if d.expected == nil {
			t.Errorf("Expected an invalid standard deviation of 0, %v", d)
		}
		if len(out) != len(d.expected) {
			t.Errorf("Expected %d elements, but got %d, %v", len(d.expected), len(out), d)
		}
		for i := 0; i < len(out); i++ {
			if math.Abs(out[i]-d.expected[i]) > 1e-7 {
				t.Errorf("Expected %v, but got %v for %v", d.expected, out, d)
				break
			}
		}
	}
}

func TestMovmeanstd(t *testing.T) {
	var err error
	var mean, std []float64

	testdata := []struct {
		data         []float64
		m            int
		expectedMean []float64
		expectedStd  []float64
	}{
		{[]float64{}, 4, nil, nil},
		{[]float64{}, 0, nil, nil},
		{[]float64{1, 1, 1, 1}, 0, nil, nil},
		{[]float64{1, 1, 1, 1}, 4, []float64{1}, []float64{0}},
		{[]float64{1, 1, 1, 1}, 2, []float64{1, 1, 1}, []float64{0, 0, 0}},
		{[]float64{-1, -1, -1, -1}, 2, []float64{-1, -1, -1}, []float64{0, 0, 0}},
		{[]float64{1, -1, -1, 1}, 2, []float64{0, -1, 0}, []float64{1, 0, 1}},
		{[]float64{1, 2, 4, 8}, 2, []float64{1.5, 3, 6}, []float64{0.5, 1, 2}},
	}

	for _, d := range testdata {
		mean, std, err = MovMeanStd(d.data, d.m)
		if err != nil {
			if d.expectedStd == nil && d.expectedMean == nil {
				// Got an error while calculating and expected an error
				continue
			} else {
				t.Errorf("Did not expect an error, %v for %v", err, d)
				break
			}
		}
		if d.expectedStd == nil {
			t.Errorf("Expected an invalid moving standard deviation, %v", d)
		}
		if len(mean) != len(d.expectedMean) {
			t.Errorf("Expected %d elements, but got %d, %v", len(d.expectedMean), len(mean), d)
		}
		for i := 0; i < len(mean); i++ {
			if math.Abs(mean[i]-d.expectedMean[i]) > 1e-7 {
				t.Errorf("Expected %v, but got %v for %v", d.expectedMean, mean, d)
				break
			}
		}

		if len(std) != len(d.expectedStd) {
			t.Errorf("Expected %d elements, but got %d, %v", len(d.expectedStd), len(std), d)
		}
		for i := 0; i < len(std); i++ {
			if math.Abs(std[i]-d.expectedStd[i]) > 1e-7 {
				t.Errorf("Expected %v, but got %v for %v", d.expectedStd, std, d)
				break
			}
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
		histo = ArcCurve(d.mpIdx)
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
		if out = Iac(d.x, d.n); out != d.expected {
			t.Errorf("Expected %.3f but got %.3f", d.expected, out)
		}
	}
}

func TestMuInvN(t *testing.T) {
	testdata := []struct {
		a           []float64
		w           int
		expectedMu  []float64
		expectedSig []float64
	}{
		{[]float64{2, 2, 2, 2, 2, 2}, 3, []float64{2, 2, 2, 2}, []float64{0, 0, 0, 0}},
		{[]float64{2, 4, 3, 5, 4, 6}, 3, []float64{3, 4, 4, 5}, []float64{math.Sqrt(2) / 2, math.Sqrt(2) / 2, math.Sqrt(2) / 2, math.Sqrt(2) / 2}},
		{[]float64{1, 1, 1, 1}, 4, []float64{1}, []float64{0}},
		{[]float64{1, 1, 1, 1}, 2, []float64{1, 1, 1}, []float64{0, 0, 0}},
		{[]float64{-1, -1, -1, -1}, 2, []float64{-1, -1, -1}, []float64{0, 0, 0}},
	}

	for _, d := range testdata {
		mu, sig := MuInvN(d.a, d.w)
		if len(mu) != len(d.expectedMu) {
			t.Errorf("Expected %d elements of mu but got %d", len(d.expectedMu), len(mu))
			continue
		}
		if len(sig) != len(d.expectedSig) {
			t.Errorf("Expected %d elements of sig but got %d", len(d.expectedSig), len(sig))
			continue
		}
		for i := 0; i < len(mu); i++ {
			if mu[i] != d.expectedMu[i] {
				t.Errorf("Expected mu: %.3f, but got %.3f", d.expectedMu, mu)
				break
			}
			if math.Abs(sig[i]-d.expectedSig[i]) > 1e-9 {
				t.Errorf("Expected sig: %.9f, but got %.9f", d.expectedSig, sig)
				break
			}
		}
	}
}

func TestBinarySplit(t *testing.T) {
	testdata := []struct {
		lb       int
		ub       int
		expected []int
	}{
		{4, 0, []int{}},
		{1, 1, []int{1}},
		{0, 1, []int{0, 1}},
		{0, 4, []int{0, 2, 1, 3, 4}},
		{0, 9, []int{0, 5, 2, 7, 1, 3, 6, 8, 4, 9}},
		{0, 16, []int{0, 8, 4, 12, 2, 6, 10, 14, 1, 3, 5, 7, 9, 11, 13, 15, 16}},
		{7, 15, []int{7, 11, 9, 13, 8, 10, 12, 14, 15}},
	}

	for _, d := range testdata {
		res := BinarySplit(d.lb, d.ub)
		if len(res) != len(d.expected) {
			t.Errorf("Expected result length of %d, but got %d for %v", len(d.expected), len(res), d)
			break
		}
		for i, v := range res {
			if v != d.expected[i] {
				t.Errorf("Expected value %d at index, %d, but got %d for %v", d.expected[i], i, v, d)
				break
			}
		}
	}
}
