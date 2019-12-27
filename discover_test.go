package matrixprofile

import (
	"math"
	"sort"
	"testing"

	"github.com/matrix-profile-foundation/go-matrixprofile/av"
)

func TestDiscoverDiscords(t *testing.T) {
	mprof := []float64{1, 2, 3, 4}
	a := []float64{1, 2, 3, 4, 5, 6}
	m := 3

	testdata := []struct {
		mp               []float64
		k                int
		exzone           int
		expectedDiscords []int
	}{
		{mprof, 4, 0, []int{3, 3, 3, 3}},
		{mprof, 4, 1, []int{3, 1}},
		{mprof, 10, 1, []int{3, 1}},
		{mprof, 0, 1, []int{}},
	}

	for _, d := range testdata {
		mp := MatrixProfile{A: a, B: a, M: m, MP: d.mp, AV: av.Default}
		discords, err := mp.DiscoverDiscords(d.k, d.exzone)
		if err != nil {
			t.Errorf("Got error %v on %v", err, d)
			return
		}
		if len(discords) != len(d.expectedDiscords) {
			t.Errorf("Got a length of %d discords, but expected %d, for %v", len(discords), len(d.expectedDiscords), d)
			return
		}
		for i, idx := range discords {
			if idx != d.expectedDiscords[i] {
				t.Errorf("expected index, %d, but got %d, for %v", d.expectedDiscords[i], idx, d)
				return
			}
		}
	}
}

func TestDiscoverMotifs(t *testing.T) {
	a := []float64{0, 0, 0.56, 0.99, 0.97, 0.75, 0, 0, 0, 0.43, 0.98, 0.99, 0.65, 0, 0, 0, 0.6, 0.97, 0.965, 0.8, 0, 0, 0}

	testdata := []struct {
		a               []float64
		b               []float64
		m               int
		k               int
		expectedMotifs  [][]int
		expectedMinDist []float64
	}{
		{
			a, nil, 7, 3,
			[][]int{{0, 14}, {0, 7}, {3, 10}},
			[]float64{0.1459619228330262, 0.3352336136782056, 0.46369664551715467},
		},
		{
			a, a, 7, 3,
			nil,
			nil,
		},
		{
			a, nil, 7, 5,
			[][]int{{0, 14}, {0, 7}, {3, 10}, {}, {}},
			[]float64{0.1459619228330262, 0.3352336136782056, 0.46369664551715467, 0, 0},
		},
		{
			[]float64{0, 1, 0, 0, 1, 0, 0}, nil, 3, 2,
			[][]int{{0, 3}, {1, 4}},
			[]float64{5.1619136559035694e-08, 0},
		},
	}

	for _, d := range testdata {
		mp, err := New(d.a, d.b, d.m)
		if err != nil {
			t.Error(err)
			return
		}

		o := NewComputeOpts()
		o.Algorithm = AlgoSTOMP
		if err = mp.Compute(o); err != nil {
			t.Error(err)
			return
		}
		motifs, err := mp.DiscoverMotifs(d.k, 2)
		if err != nil {
			if d.expectedMotifs == nil {
				continue
			}
			t.Error(err)
			return
		}

		for i := range motifs {
			sort.Ints(motifs[i].Idx)
		}

		for i, mg := range motifs {
			if len(mg.Idx) != len(d.expectedMotifs[i]) {
				t.Errorf("expected %d motifs for group %d, but got %d, %v, for %v", len(d.expectedMotifs[i]), i, len(mg.Idx), mg.Idx, d)
				return
			}

			for j, idx := range mg.Idx {
				if idx != d.expectedMotifs[i][j] {
					t.Errorf("expected index, %d for group %d, but got %d for %v", d.expectedMotifs[i][j], i, idx, d)
					return
				}
			}
			if math.Abs(mg.MinDist-d.expectedMinDist[i]) > 1e-7 {
				t.Errorf("expected minimum distance, %v for group %d, but got %v for %v", d.expectedMinDist[i], i, mg.MinDist, d)
				return
			}
		}
	}
}

func TestDiscoverSegments(t *testing.T) {
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
		mp := MatrixProfile{Idx: d.mpIdx}
		minIdx, minVal, histo = mp.DiscoverSegments()
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
