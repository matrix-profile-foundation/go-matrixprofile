package matrixprofile

import (
	"math"
	"os"
	"testing"
)

func TestPMPSave(t *testing.T) {
	ts := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}
	p, err := NewPMP(ts, nil)
	p.Compute(NewPMPOpts(3, 5))
	filepath := "./mp.json"
	err = p.Save(filepath, "json")
	if err != nil {
		t.Errorf("Received error while saving matrix profile, %v", err)
	}
	if err = os.Remove(filepath); err != nil {
		t.Errorf("Could not remove file, %s, %v", filepath, err)
	}
}

func TestPMPLoad(t *testing.T) {
	ts := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}
	p, err := NewPMP(ts, nil)
	p.Compute(NewPMPOpts(3, 5))
	filepath := "./mp.json"
	if err = p.Save(filepath, "json"); err != nil {
		t.Errorf("Received error while saving matrix profile, %v", err)
	}

	newP := &PMP{}
	if err = newP.Load(filepath, "json"); err != nil {
		t.Errorf("Failed to load %s, %v", filepath, err)
	}

	if err = os.Remove(filepath); err != nil {
		t.Errorf("Could not remove file, %s, %v", filepath, err)
	}

	if len(newP.A) != len(ts) {
		t.Errorf("Expected timeseries length of %d, but got %d", len(ts), len(newP.A))
	}

}

func TestComputePmp(t *testing.T) {
	var err error
	var p *PMP

	testdata := []struct {
		a            []float64
		b            []float64
		lb           int
		ub           int
		p            int
		expectedPMP  [][]float64
		expectedPIdx [][]int
	}{
		{[]float64{}, []float64{}, 2, 2, 1, nil, nil},
		{[]float64{1, 1, 1, 1, 1}, []float64{}, 2, 2, 1, nil, nil},
		{[]float64{}, []float64{1, 1, 1, 1, 1}, 2, 2, 1, nil, nil},
		{[]float64{1, 2, 1, 3, 1}, []float64{2, 1, 1, 2, 1, 3, 1, -1, -2}, 2, 2, 1, [][]float64{{0, 0, 0, 0}}, [][]int{{2, 3, 2, 3}}},
		{[]float64{1, 1, 1, 1, 1}, []float64{1, 1, 1, 1, 1, 2, 2, 3, 4, 5}, 2, 2, 1, [][]float64{{2, 2, 2, 2}}, [][]int{{0, 1, 2, 3}}},
		{[]float64{0, 0.99, 1, 0, 0, 0.98, 1, 0, 0, 0.96, 1, 0}, []float64{0, 0.99, 1, 0, 0, 0.98, 1, 0, 0, 0.96, 1, 0}, 4, 4, 1,
			[][]float64{{0, 0, 0, 0, 0, 0, 0, 0, 0}},
			[][]int{{0, 1, 2, 3, 4, 5, 6, 7, 8}}},
		{[]float64{0, 1, 1, 1, 0, 0, 2, 1, 0, 0, 2, 1}, nil, 4, 4, 1,
			[][]float64{{1.9550, 1.8388, 0.8739, 0, 0, 1.9550, 0.8739, 0, 0}},
			[][]int{{4, 2, 6, 7, 8, 1, 2, 3, 4}}},
		{[]float64{0, 0.99, 1, 0, 0, 0.98, 1, 0, 0, 0.96, 1, 0}, nil, 4, 4, 1,
			[][]float64{{0.014355, 0.014355, 0.029138, 0.029138, 0.014355, 0.014355, 0.029138, 0.029138, 0.029138}},
			[][]int{{4, 5, 6, 7, 0, 1, 2, 3, 4}}},
		{[]float64{0, 0.99, 1, 0, 0, 0.98, 1, 0, 0, 0.96, 1, 0}, nil, 4, 4, 2,
			[][]float64{{0.014355, 0.014355, 0.029138, 0.029138, 0.014355, 0.014355, 0.029138, 0.029138, 0.029138}},
			[][]int{{4, 5, 6, 7, 0, 1, 2, 3, 4}}},
		{[]float64{0, 0.99, 1, 0, 0, 0.98, 1, 0, 0, 0.96, 1, 0}, nil, 4, 4, 4,
			[][]float64{{0.014355, 0.014355, 0.029138, 0.029138, 0.014355, 0.014355, 0.029138, 0.029138, 0.029138}},
			[][]int{{4, 5, 6, 7, 0, 1, 2, 3, 4}}},
		{[]float64{0, 0.99, 1, 0, 0, 0.98, 1, 0, 0, 0.96, 1, 0}, nil, 4, 4, 100,
			[][]float64{{0.014355, 0.014355, 0.029138, 0.029138, 0.014355, 0.014355, 0.029138, 0.029138, 0.029138}},
			[][]int{{4, 5, 6, 7, 0, 1, 2, 3, 4}}},
		{[]float64{0, 0.99, 1, 0, 0, 0.98, 1, 0, 0, 0.96, 1, 0}, nil, 3, 5, 1,
			[][]float64{
				{0.015225, 0.015225, 0.000000, 0.000000, 0.015225, 0.015225, 0.000000, 0.000000, 0.030899, 0.030899},
				{0.014355, 0.014355, 0.029138, 0.029138, 0.014355, 0.014355, 0.029138, 0.029138, 0.029138},
				{0.014651, 0.029742, 0.033992, 0.029742, 0.014651, 0.029742, 0.033992, 0.029742},
			},
			[][]int{
				{4, 5, 6, 7, 0, 1, 2, 3, 4, 5},
				{4, 5, 6, 7, 0, 1, 2, 3, 4},
				{4, 5, 6, 7, 0, 1, 2, 3},
			}},
	}

	for _, d := range testdata {
		p, err = NewPMP(d.a, d.b)
		if err != nil {
			if d.expectedPMP == nil {
				// Got an error while creating a new matrix profile
				continue
			} else {
				t.Errorf("Did not expect an error, %v,  while creating new mp for %v", err, d)
				return
			}
		}

		o := NewPMPOpts(d.lb, d.ub)
		o.MPOpts.Parallelism = d.p
		err = p.Compute(o)
		if err != nil {
			if d.expectedPMP == nil {
				// Got an error while z normalizing and expected an error
				continue
			} else {
				t.Errorf("Did not expect an error, %v, while calculating for %v", err, d)
				break
			}
		}
		if d.expectedPMP == nil {
			t.Errorf("Expected an invalid calculation, %+v", d)
			break
		}

		if len(p.PMP) != len(d.expectedPMP) {
			t.Errorf("Expected %d elements, but got %d, %+v", len(d.expectedPMP), len(p.PMP), d)
			return
		}
		for j := 0; j < len(p.PMP); j++ {
			if len(p.PMP[j]) != len(d.expectedPMP[j]) {
				t.Errorf("Expected %d elements, but got %d, %+v", len(d.expectedPMP[j]), len(p.PMP[j]), d)
				return
			}
			for i := 0; i < len(p.PMP[j]); i++ {
				if math.Abs(p.PMP[j][i]-d.expectedPMP[j][i]) > 1e-4 {
					t.Errorf("Expected\n%.6f, but got\n%.6f for\n%+v", d.expectedPMP[j], p.PMP[j], d)
					break
				}
			}
			for i := 0; i < len(p.PIdx[j]); i++ {
				if math.Abs(float64(p.PIdx[j][i]-d.expectedPIdx[j][i])) > 1e-7 {
					t.Errorf("Expected %d,\nbut got\n%v for\n%+v", d.expectedPIdx[j], p.PIdx[j], d)
					break
				}
			}
		}
	}
}
