package matrixprofile

import (
	"math"
	"os"
	"testing"

	"gonum.org/v1/gonum/fourier"
)

func TestNewKMP(t *testing.T) {
	testdata := []struct {
		t           [][]float64
		w           int
		expectedErr bool
	}{
		{[][]float64{}, 2, true},
		{[][]float64{{1, 1, 1, 1, 1}}, 2, false},
		{[][]float64{{1, 1, 1, 1, 1}}, 1, true},
		{[][]float64{{1, 1, 1, 1, 1}}, 6, true},
		{[][]float64{{1, 1, 1, 1, 1}, {1, 1, 1}}, 2, true},
	}

	for _, d := range testdata {
		_, err := NewKMP(d.t, d.w)
		if d.expectedErr && err == nil {
			t.Errorf("Expected an error, but got none for %v", d)
		}
		if !d.expectedErr && err != nil {
			t.Errorf("Expected no error, but got %v for %v", err, d)
		}
	}
}

func TestKCrossCorrelate(t *testing.T) {
	var err error
	var mp *KMP

	testdata := []struct {
		t        [][]float64
		w        int
		expected [][]float64
	}{
		{[][]float64{{1, 1, 1, 1, 1}}, 2, [][]float64{{2, 2, 2, 2}}},
		{[][]float64{{1, 2, 3, 3, 2, 1}}, 2, [][]float64{{5, 8, 9, 7, 4}}},
		{[][]float64{{1, 2, 3, 3, 2, 1, 1}}, 2, [][]float64{{5, 8, 9, 7, 4, 3}}},
		{[][]float64{
			{1, 2, 3, 3, 2, 1, 1},
			{2, 4, 3, 3, 2, 1, 1},
		}, 2,
			[][]float64{
				{5, 8, 9, 7, 4, 3},
				{20, 20, 18, 14, 8, 6},
			}},
	}

	for _, d := range testdata {
		mp, err = NewKMP(d.t, d.w)
		if err != nil {
			if d.expected == nil {
				// Got an error while creating a new matrix profile
				continue
			} else {
				t.Errorf("did not expect to get an error , %v, for %v", err, d)
			}
		}

		fft := fourier.NewFFT(mp.n)
		D := make([][]float64, len(mp.T))
		mp.crossCorrelate(0, fft, D)
		if err != nil && d.expected == nil {
			// Got an error while z normalizing and expected an error
			continue
		}
		if d.expected == nil {
			t.Errorf("Expected an invalid cross correlation calculation, %v", d)
		}
		if err != nil {
			t.Errorf("Did not expect error, %v", err)
		}
		if len(D) != len(d.expected) {
			t.Errorf("Expected %d dimensions, but got %d, %v", len(d.expected), len(D), d)
		}
		for i := 0; i < len(D); i++ {
			for j := 0; j < len(D[0]); j++ {
				if math.Abs(D[i][j]-d.expected[i][j]) > 1e-7 {
					t.Errorf("Expected %v, but got %v for %v", d.expected, D, d)
					break
				}
			}
		}

	}
}

func TestColumnWiseSort(t *testing.T) {
	testdata := []struct {
		d         [][]float64
		expectedD [][]float64
	}{
		{
			[][]float64{
				{1, 4, 9},
				{2, 6, 4},
				{3, 2, 3},
				{4, 1, 2}},
			[][]float64{
				{1, 1, 2},
				{2, 2, 3},
				{3, 4, 4},
				{4, 6, 9}},
		},
	}

	for _, d := range testdata {
		mp := &KMP{W: 5, n: 7}
		mp.columnWiseSort(d.d)

		if len(d.d) != len(d.expectedD) {
			t.Errorf("Expected %d dimensions, but got %d, %+v", len(d.expectedD), len(d.d), d)
			break
		}
		for dim := 0; dim < len(d.d); dim++ {
			for i := 0; i < mp.n-mp.W-1; i++ {
				if math.Abs(d.d[dim][i]-d.expectedD[dim][i]) > 1e-7 {
					t.Errorf("Expected\n%.4f, but got\n%.4f for\n%+v", d.expectedD[dim], d.d[dim], d)
					break
				}
			}
		}
	}
}

func TestMStomp(t *testing.T) {
	var err error
	var mp *KMP

	testdata := []struct {
		t          [][]float64
		m          int
		expectedMP [][]float64
	}{
		{
			[][]float64{
				{0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0},
				{0, 0, -1, -1, 0, 0, 0, -1, -1, 0, 0},
				{0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0}},
			4,
			[][]float64{
				{0, 0, 0, 1.838803373524, 1.838803373524, 0, 0, 0},
				{0, 0, 0, 1.838803373524, 1.838803373524, 0, 0, 0},
				{1.184098445303, 1.184098445303, 1.184098445303, 1.995669274602, 2.409967360985, 1.184098445303, 1.184098445303, 1.184098445303}},
		},
	}

	for _, d := range testdata {
		mp, err = NewKMP(d.t, d.m)
		if err != nil {
			if d.expectedMP == nil {
				// Got an error while creating a new matrix profile
				continue
			} else {
				t.Errorf("Did not expect an error, %v,  while creating new mp for %v", err, d)
			}
		}

		err = mp.Compute()
		if err != nil {
			if d.expectedMP == nil {
				// Got an error while z normalizing and expected an error
				continue
			} else {
				t.Errorf("Did not expect an error, %v, while calculating stomp for %v", err, d)
				break
			}
		}
		if d.expectedMP == nil {
			t.Errorf("Expected an invalid STOMP calculation, %+v", d)
			break
		}

		if len(mp.MP) != len(d.expectedMP) {
			t.Errorf("Expected %d dimensions, but got %d, %+v", len(d.expectedMP), len(mp.MP), d)
		}
		for dim := 0; dim < len(d.t); dim++ {
			for i := 0; i < mp.n-mp.W-1; i++ {
				if math.Abs(mp.MP[dim][i]-d.expectedMP[dim][i]) > 1e-7 {
					for dd := 0; dd < len(d.t); dd++ {
						t.Errorf("Expected\n%.12f, but got\n%.12f for\n%+v", d.expectedMP[dd], mp.MP[dd], d)
					}
					break
				}
			}
		}
	}
}

func TestKMPSave(t *testing.T) {
	ts := [][]float64{{1, 2, 3, 4, 5, 6, 7, 8, 9}}
	m := 3
	p, err := NewKMP(ts, m)
	p.Compute()
	filepath := "./kmp.json"
	err = p.Save(filepath, "json")
	if err != nil {
		t.Errorf("Received error while saving matrix profile, %v", err)
	}
	if err = os.Remove(filepath); err != nil {
		t.Errorf("Could not remove file, %s, %v", filepath, err)
	}
}

func TestKMPLoad(t *testing.T) {
	ts := [][]float64{{1, 2, 3, 4, 5, 6, 7, 8, 9}}
	w := 3
	p, err := NewKMP(ts, w)
	p.Compute()
	filepath := "./kmp.json"
	if err = p.Save(filepath, "json"); err != nil {
		t.Errorf("Received error while saving matrix profile, %v", err)
	}

	newP := &KMP{}
	if err = newP.Load(filepath, "json"); err != nil {
		t.Errorf("Failed to load %s, %v", filepath, err)
	}

	if err = os.Remove(filepath); err != nil {
		t.Errorf("Could not remove file, %s, %v", filepath, err)
	}

	if newP.W != w {
		t.Errorf("Expected window of %d, but got %d", w, newP.W)
	}
	if len(newP.T) != len(ts) {
		t.Errorf("Expected timeseries length of %d, but got %d", len(ts), len(newP.T))
	}

}
