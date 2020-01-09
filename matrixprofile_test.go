package matrixprofile

import (
	"math"
	"os"
	"sort"
	"testing"

	"github.com/matrix-profile-foundation/go-matrixprofile/av"
	"gonum.org/v1/gonum/fourier"
)

func TestNew(t *testing.T) {
	testdata := []struct {
		a           []float64
		b           []float64
		m           int
		expectedErr bool
	}{
		{[]float64{}, []float64{}, 2, true},
		{[]float64{1, 1, 1, 1, 1}, []float64{}, 2, true},
		{[]float64{1, 1, 1, 1, 1}, nil, 2, false},
		{[]float64{1, 1, 1, 1, 1}, nil, 6, true},
		{[]float64{1, 1}, []float64{1, 1, 1, 1, 1, 1, 1, 1}, 3, true},
		{[]float64{}, []float64{1, 1, 1, 1, 1}, 2, true},
		{[]float64{1, 2, 3, 4, 5}, []float64{1, 1, 1, 1, 1}, 2, false},
		{[]float64{1, 2, 3, 4, 5}, []float64{1, 1, 1, 1, 1}, 1, true},
		{[]float64{1, 2, 3, 4, 5}, []float64{1, 1, 1, 1, 1}, 4, false},
	}

	for _, d := range testdata {
		_, err := New(d.a, d.b, d.m)
		if d.expectedErr && err == nil {
			t.Errorf("Expected an error, but got none for %v", d)
			return
		}
		if !d.expectedErr && err != nil {
			t.Errorf("Expected no error, but got %v for %v", err, d)
			return
		}
	}
}

func TestApplyAV(t *testing.T) {
	mprof := []float64{4, 6, 10, 2, 1, 0, 1, 2, 0, 0, 1, 2, 6}

	testdata := []struct {
		b          []float64
		w          int
		av         av.AV
		expectedMP []float64
	}{
		{[]float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}, 4, av.Default, mprof},
	}

	var mp MatrixProfile
	var err error
	var out []float64
	for _, d := range testdata {
		newMP := make([]float64, len(mprof))
		copy(newMP, mprof)
		mp = MatrixProfile{B: d.b, W: d.w, MP: newMP, AV: d.av}
		out, err = mp.ApplyAV()
		if err != nil {
			t.Fatal(err)
		}

		if len(out) != len(d.expectedMP) {
			t.Errorf("Expected %d elements, but got %d, %+v", len(d.expectedMP), len(out), d)
			break
		}
		for i := 0; i < len(out); i++ {
			if math.Abs(float64(out[i]-d.expectedMP[i])) > 1e-7 {
				t.Errorf("Expected %v,\nbut got\n%v for %+v", d.expectedMP, out, d)
				break
			}
		}
	}
}

func TestSave(t *testing.T) {
	ts := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9}
	w := 3
	p, err := New(ts, nil, w)
	p.Compute(NewMPOpts())
	filepath := "./mp.json"
	err = p.Save(filepath, "json")
	if err != nil {
		t.Errorf("Received error while saving matrix profile, %v", err)
	}
	if err = os.Remove(filepath); err != nil {
		t.Errorf("Could not remove file, %s, %v", filepath, err)
	}
}

func TestLoad(t *testing.T) {
	ts := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9}
	w := 3
	p, err := New(ts, nil, w)
	p.Compute(NewMPOpts())
	filepath := "./mp.json"
	if err = p.Save(filepath, "json"); err != nil {
		t.Errorf("Received error while saving matrix profile, %v", err)
	}

	newP := &MatrixProfile{}
	if err = newP.Load(filepath, "json"); err != nil {
		t.Errorf("Failed to load %s, %v", filepath, err)
	}

	if err = os.Remove(filepath); err != nil {
		t.Errorf("Could not remove file, %s, %v", filepath, err)
	}

	if newP.W != w {
		t.Errorf("Expected window of %d, but got %d", w, newP.W)
	}
	if len(newP.A) != len(ts) {
		t.Errorf("Expected timeseries length of %d, but got %d", len(ts), len(newP.A))
	}

}

func TestMPDist(t *testing.T) {
	testData := []struct {
		a        []float64
		b        []float64
		m        int
		expected float64
	}{
		{
			[]float64{1, 2, 3, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
			[]float64{0, 0, 0, 0, 0, 0, 0, 0, -1, -2, -3, -2, -1, 0, 1, 2, 1, 0},
			5,
			0,
		},
		{
			[]float64{1, 2, 3, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
			[]float64{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 0},
			5,
			0,
		},
	}
	for _, d := range testData {
		res, err := MPDist(d.a, d.b, d.m, nil)
		if err != nil {
			t.Errorf("Did not expect to get an error, %v", err)
		}
		if math.Abs(res-d.expected) > 1e-4 {
			t.Errorf("Expected %.6f, but got %.6f", d.expected, res)
		}
	}
}

func TestCrossCorrelate(t *testing.T) {
	var err error
	var out []float64
	var mp *MatrixProfile

	testdata := []struct {
		q        []float64
		t        []float64
		expected []float64
	}{
		{[]float64{1, 1}, []float64{1, 1, 1, 1, 1}, []float64{2, 2, 2, 2}},
		{[]float64{1, 2}, []float64{1, 2, 3, 3, 2, 1}, []float64{5, 8, 9, 7, 4}},
		{[]float64{1, 2}, []float64{1, 2, 3, 3, 2, 1, 1}, []float64{5, 8, 9, 7, 4, 3}},
		{[]float64{1, 2, 1}, []float64{1, 2, 3, 4, 3, 2, 1}, []float64{8, 12, 14, 12, 8}},
		{[]float64{1, 2, 1}, []float64{1, 2, 3, 4, 3, 2, 1, 1}, []float64{8, 12, 14, 12, 8, 5}},
	}

	for _, d := range testdata {
		mp, err = New(d.q, d.t, len(d.q))
		if err != nil {
			if d.expected == nil {
				// Got an error while creating a new matrix profile
				continue
			} else {
				t.Errorf("did not expect to get an error , %v, for %v", err, d)
				return
			}
		}
		if err = mp.initCaches(); err != nil {
			t.Errorf("Failed to initialize cache, %v", err)
		}

		fft := fourier.NewFFT(mp.N)
		out = mp.crossCorrelate(d.q, fft)
		if err != nil && d.expected == nil {
			// Got an error while z normalizing and expected an error
			continue
		}
		if d.expected == nil {
			t.Errorf("Expected an invalid cross correlation calculation, %v", d)
			return
		}
		if err != nil {
			t.Errorf("Did not expect error, %v", err)
			return
		}
		if len(out) != len(d.expected) {
			t.Errorf("Expected %d elements, but got %d, %v", len(d.expected), len(out), d)
			return
		}
		for i := 0; i < len(out); i++ {
			if math.Abs(out[i]-d.expected[i]) > 1e-7 {
				t.Errorf("Expected %v, but got %v for %v", d.expected, out, d)
				break
			}
		}

	}
}

func TestMass(t *testing.T) {
	var err error
	var mp *MatrixProfile
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
		{[]float64{0, 1, 1, 0}, []float64{0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0}, []float64{0, 2.8284271247461903, 4, 2.8284271247461903, 0, 2.82842712474619, 4, 2.8284271247461903, 0}},
		{[]float64{0, 1, 1, 0}, []float64{1e-6, 1e-5, 1e-5, 1e-5, 5, 5, 1e-5, 1e-5, 1e-5, 1e-5, 7, 7, 1e-5, 1e-5},
			[]float64{1.838803373328544, 3.552295335908461, 2.828427124746192, 6.664001874625056e-08, 2.8284271247461885,
				3.5522953359084606, 2.8284271366321914, 3.5522953359084606, 2.82842712474619, 0, 2.82842712474619070}},
	}

	for _, d := range testdata {
		mp, err = New(d.q, d.t, len(d.q))
		if err != nil && d.expected == nil {
			// Got an error while creating a new matrix profile
			continue
		}
		if err = mp.initCaches(); err != nil {
			t.Errorf("Failed to initialize cache, %v", err)
		}
		out = make([]float64, mp.N-mp.W+1)
		fft := fourier.NewFFT(mp.N)
		err = mp.mass(d.q, out, fft)
		if err != nil && d.expected == nil {
			// Got an error while z normalizing and expected an error
			continue
		}
		if d.expected == nil {
			t.Errorf("Expected an invalid mass calculation, %v", d)
			return
		}
		if err != nil {
			t.Errorf("Did not expect error, %v", err)
			return
		}
		if len(out) != len(d.expected) {
			t.Errorf("Expected %d elements, but got %d, %v", len(d.expected), len(out), d)
			return
		}
		for i := 0; i < len(out); i++ {
			if math.IsNaN(out[i]) {
				t.Errorf("Got NaN in output, %v", out)
				break
			}
			if math.Abs(out[i]-d.expected[i]) > 1e-7 {
				t.Errorf("Expected %v\n, but got %v\nfor %v", d.expected, out, d)
				break
			}
		}
	}
}

func TestDistanceProfile(t *testing.T) {
	var err error
	var mprof []float64
	var mp *MatrixProfile

	testdata := []struct {
		q          []float64
		t          []float64
		m          int
		idx        int
		expectedMP []float64
	}{
		{[]float64{}, []float64{}, 2, 0, nil},
		{[]float64{1, 1, 1, 1, 1}, []float64{}, 2, 0, nil},
		{[]float64{}, []float64{1, 1, 1, 1, 1}, 2, 0, nil},
		{[]float64{0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0}, nil, 4, 0, []float64{math.Inf(1), math.Inf(1), 4, 2.8284271247461903, 0, 2.8284271247461903, 4, 2.8284271247461903, 0}},
		{[]float64{0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0}, nil, 4, 9, nil},
	}

	for _, d := range testdata {
		mp, err = New(d.q, d.t, d.m)
		if err != nil && d.expectedMP == nil {
			// Got an error while creating a new matrix profile
			continue
		}

		if err = mp.initCaches(); err != nil {
			t.Errorf("Failed to initialize cache, %v", err)
		}

		mprof = make([]float64, mp.N-mp.W+1)
		fft := fourier.NewFFT(mp.N)
		err = mp.distanceProfile(d.idx, mprof, fft)
		if err != nil && d.expectedMP == nil {
			// Got an error while z normalizing and expected an error
			continue
		}
		if d.expectedMP == nil {
			t.Errorf("Expected an invalid distance profile calculation, %+v", d)
			return
		}
		if err != nil {
			t.Errorf("Did not expect error, %v\n%+v", err, d)
			return
		}
		if len(mprof) != len(d.expectedMP) {
			t.Errorf("Expected %d elements, but got %d\n%+v", len(d.expectedMP), len(mprof), d)
			return
		}
		for i := 0; i < len(mprof); i++ {
			if math.Abs(mprof[i]-d.expectedMP[i]) > 1e-7 {
				t.Errorf("Expected\n%.7f, but got\n%.7f for\n%+v", d.expectedMP, mprof, d)
				break
			}
		}
	}
}

func TestCalculateDistanceProfile(t *testing.T) {
	var err error
	var mprof []float64
	var mp *MatrixProfile

	testdata := []struct {
		q          []float64
		t          []float64
		m          int
		idx        int
		expectedMP []float64
	}{
		{[]float64{}, []float64{}, 2, 0, nil},
		{[]float64{1, 1, 1, 1, 1}, []float64{}, 2, 0, nil},
		{[]float64{}, []float64{1, 1, 1, 1, 1}, 2, 0, nil},
		{[]float64{0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0}, nil, 4, 0, []float64{math.Inf(1), math.Inf(1), 4, 2.8284271247461903, 0, 2.8284271247461903, 4, 2.8284271247461903, 0}},
		{[]float64{0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0}, nil, 4, 9, nil},
	}

	for _, d := range testdata {
		mp, err = New(d.q, d.t, d.m)
		if err != nil && d.expectedMP == nil {
			// Got an error while creating a new matrix profile
			continue
		}

		if err = mp.initCaches(); err != nil {
			t.Errorf("Failed to initialize cache, %v", err)
		}

		fft := fourier.NewFFT(mp.N)
		dot := mp.crossCorrelate(mp.A[:mp.W], fft)

		mprof = make([]float64, mp.N-mp.W+1)
		err = mp.calculateDistanceProfile(dot, d.idx, mprof)
		if err != nil {
			if d.expectedMP == nil {
				// Got an error while z normalizing and expected an error
				continue
			} else {
				t.Errorf("Did not expect to get error, %v, for %v", err, d)
				return
			}
		}
		if d.expectedMP == nil {
			t.Errorf("Expected an invalid distance profile calculation, %+v", d)
			return
		}
		if err != nil {
			t.Errorf("Did not expect error, %v\n%+v", err, d)
			return
		}
		if len(mprof) != len(d.expectedMP) {
			t.Errorf("Expected %d elements, but got %d\n%+v", len(d.expectedMP), len(mprof), d)
			return
		}
		for i := 0; i < len(mprof); i++ {
			if math.Abs(mprof[i]-d.expectedMP[i]) > 1e-7 {
				t.Errorf("Expected\n%.7f, but got\n%.7f for\n%+v", d.expectedMP, mprof, d)
				break
			}
		}
	}

}

func TestComputeStmp(t *testing.T) {
	var err error
	var mp *MatrixProfile

	testdata := []struct {
		q             []float64
		t             []float64
		m             int
		expectedMP    []float64
		expectedMPIdx []int
	}{
		{[]float64{}, []float64{}, 2, nil, nil},
		{[]float64{1, 1, 1, 1, 1}, []float64{}, 2, nil, nil},
		{[]float64{}, []float64{1, 1, 1, 1, 1}, 2, nil, nil},
		{[]float64{1, 1}, []float64{1, 1, 1, 1, 1}, 2, nil, nil},
		{[]float64{0, 0.99, 1, 0, 0, 0.98, 1, 0, 0, 0.96, 1, 0}, nil, 4,
			[]float64{0.014355034678331376, 0.014355034678269504, 0.0291386974835963, 0.029138697483626783, 0.01435503467830044, 0.014355034678393249, 0.029138697483504856, 0.029138697483474377, 0.0291386974835963},
			[]int{4, 5, 6, 7, 0, 1, 2, 3, 4}},
	}

	for _, d := range testdata {
		mp, err = New(d.q, d.t, d.m)
		if err != nil && d.expectedMP == nil {
			// Got an error while creating a new matrix profile
			continue
		}

		o := NewMPOpts()
		o.Algorithm = AlgoSTMP

		err = mp.Compute(o)
		if err != nil && d.expectedMP == nil {
			// Got an error while z normalizing and expected an error
			continue
		}
		if d.expectedMP == nil {
			t.Errorf("Expected an invalid STMP calculation, %+v", d)
			return
		}
		if err != nil {
			t.Errorf("Did not expect error, %v, %+v", err, d)
			return
		}
		if len(mp.MP) != len(d.expectedMP) {
			t.Errorf("Expected %d elements, but got %d, %+v", len(d.expectedMP), len(mp.MP), d)
			return
		}
		for i := 0; i < len(mp.MP); i++ {
			if math.Abs(mp.MP[i]-d.expectedMP[i]) > 1e-7 {
				t.Errorf("Expected\n%v, but got\n%v for\n%+v", d.expectedMP, mp.MP, d)
				break
			}
		}
		for i := 0; i < len(mp.Idx); i++ {
			if math.Abs(float64(mp.Idx[i]-d.expectedMPIdx[i])) > 1e-7 {
				t.Errorf("Expected %v,\nbut got\n%v for\n%+v", d.expectedMPIdx, mp.Idx, d)
				break
			}
		}
	}
}

func TestComputeStamp(t *testing.T) {
	var err error
	var mp *MatrixProfile

	testdata := []struct {
		q             []float64
		t             []float64
		m             int
		sample        float64
		expectedMP    []float64
		expectedMPIdx []int
	}{
		{[]float64{}, []float64{}, 2, 1.0, nil, nil},
		{[]float64{1, 1, 1, 1, 1}, []float64{}, 2, 1.0, nil, nil},
		{[]float64{}, []float64{1, 1, 1, 1, 1}, 2, 1.0, nil, nil},
		{[]float64{1, 1}, []float64{1, 1, 1, 1, 1}, 2, 1.0, nil, nil},
		{[]float64{0, 0.99, 1, 0, 0, 0.98, 1, 0, 0, 0.96, 1, 0}, nil, 4, 1.0,
			[]float64{0.014355034678331376, 0.014355034678269504, 0.0291386974835963, 0.029138697483626783, 0.01435503467830044, 0.014355034678393249, 0.029138697483504856, 0.029138697483474377, 0.0291386974835963},
			[]int{4, 5, 6, 7, 0, 1, 2, 3, 4}},
		{[]float64{0, 0.99, 1, 0, 0, 0.98, 1, 0, 0, 0.96, 1, 0}, nil, 4, 0.0, nil, nil},
	}

	for _, d := range testdata {
		mp, err = New(d.q, d.t, d.m)
		if err != nil && d.expectedMP == nil {
			// Got an error while creating a new matrix profile
			continue
		}

		o := NewMPOpts()
		o.Algorithm = AlgoSTAMP
		o.Sample = d.sample

		err = mp.Compute(o)

		if err != nil && d.expectedMP == nil {
			// Got an error while z normalizing and expected an error
			continue
		}
		if d.expectedMP == nil {
			t.Errorf("Expected an invalid STAMP calculation, %+v", d)
			return
		}
		if err != nil {
			t.Errorf("Did not expect error, %v, %+v", err, d)
			return
		}
		if len(mp.MP) != len(d.expectedMP) {
			t.Errorf("Expected %d elements, but got %d, %+v", len(d.expectedMP), len(mp.MP), d)
			return
		}
		for i := 0; i < len(mp.MP); i++ {
			if math.Abs(mp.MP[i]-d.expectedMP[i]) > 1e-7 {
				t.Errorf("Expected\n%v, but got\n%v for\n%+v", d.expectedMP, mp.MP, d)
				break
			}
		}
		for i := 0; i < len(mp.Idx); i++ {
			if math.Abs(float64(mp.Idx[i]-d.expectedMPIdx[i])) > 1e-7 {
				t.Errorf("Expected %v,\nbut got\n%v for\n%+v", d.expectedMPIdx, mp.Idx, d)
				break
			}
		}

	}
}

func TestComputeStomp(t *testing.T) {
	var err error
	var mp *MatrixProfile

	testdata := []struct {
		q             []float64
		t             []float64
		m             int
		p             int
		expectedMP    []float64
		expectedMPIdx []int
	}{
		{[]float64{}, []float64{}, 2, 1, nil, nil},
		{[]float64{1, 1, 1, 1, 1}, []float64{}, 2, 1, nil, nil},
		{[]float64{}, []float64{1, 1, 1, 1, 1}, 2, 1, nil, nil},
		{[]float64{1, 1}, []float64{1, 1, 1, 1, 1}, 2, 1, []float64{math.Inf(1), math.Inf(1), math.Inf(1), math.Inf(1)}, []int{math.MaxInt64, math.MaxInt64, math.MaxInt64, math.MaxInt64}},
		{[]float64{1, 1, 1, 1, 1, 1, 1, 1}, []float64{1, 1, 1, 1, 1}, 2, 1, []float64{math.Inf(1), math.Inf(1), math.Inf(1), math.Inf(1)}, []int{math.MaxInt64, math.MaxInt64, math.MaxInt64, math.MaxInt64}},
		{[]float64{0, 0.99, 1, 0, 0, 0.98, 1, 0, 0, 0.96, 1, 0}, []float64{0, 0.99, 1, 0, 0, 0.98, 1, 0, 0, 0.96, 1, 0}, 4, 1,
			[]float64{0, 0, 0, 0, 0, 0, 0, 0, 0},
			[]int{0, 1, 2, 3, 4, 5, 6, 7, 8}},
		{[]float64{0, 0.99, 1, 0, 0, 0.98, 1, 0, 0, 0.96, 1, 0}, nil, 4, 1,
			[]float64{0.014355034678331376, 0.014355034678269504, 0.0291386974835963, 0.029138697483626783, 0.01435503467830044, 0.014355034678393249, 0.029138697483504856, 0.029138697483474377, 0.0291386974835963},
			[]int{4, 5, 6, 7, 0, 1, 2, 3, 4}},
		{[]float64{0, 0.99, 1, 0, 0, 0.98, 1, 0, 0, 0.96, 1, 0}, nil, 4, 2,
			[]float64{0.014355034678331376, 0.014355034678269504, 0.0291386974835963, 0.029138697483626783, 0.01435503467830044, 0.014355034678393249, 0.029138697483504856, 0.029138697483474377, 0.0291386974835963},
			[]int{4, 5, 6, 7, 0, 1, 2, 3, 4}},
		{[]float64{0, 0.99, 1, 0, 0, 0.98, 1, 0, 0, 0.96, 1, 0}, nil, 4, 4,
			[]float64{0.014355034678331376, 0.014355034678269504, 0.0291386974835963, 0.029138697483626783, 0.01435503467830044, 0.014355034678393249, 0.029138697483504856, 0.029138697483474377, 0.0291386974835963},
			[]int{4, 5, 6, 7, 0, 1, 2, 3, 4}},
		{[]float64{0, 0.99, 1, 0, 0, 0.98, 1, 0, 0, 0.96, 1, 0}, nil, 4, 100,
			[]float64{0.014355034678331376, 0.014355034678269504, 0.0291386974835963, 0.029138697483626783, 0.01435503467830044, 0.014355034678393249, 0.029138697483504856, 0.029138697483474377, 0.0291386974835963},
			[]int{4, 5, 6, 7, 0, 1, 2, 3, 4}},
	}

	for _, d := range testdata {
		mp, err = New(d.q, d.t, d.m)
		if err != nil {
			if d.expectedMP == nil {
				// Got an error while creating a new matrix profile
				continue
			} else {
				t.Errorf("Did not expect an error, %v,  while creating new mp for %v", err, d)
				return
			}
		}

		o := NewMPOpts()
		o.Algorithm = AlgoSTOMP
		err = mp.Compute(o)
		if err != nil {
			if d.expectedMP == nil {
				// Got an error while z normalizing and expected an error
				continue
			} else {
				t.Errorf("Did not expect an error, %v, while calculating for %v", err, d)
				break
			}
		}
		if d.expectedMP == nil {
			t.Errorf("Expected an invalid STOMP calculation, %+v", d)
			break
		}

		if len(mp.MP) != len(d.expectedMP) {
			t.Errorf("Expected %d elements, but got %d, %+v", len(d.expectedMP), len(mp.MP), d)
			return
		}
		for i := 0; i < len(mp.MP); i++ {
			if math.Abs(mp.MP[i]-d.expectedMP[i]) > 1e-7 {
				t.Errorf("Expected\n%.4f, but got\n%.4f for\n%+v", d.expectedMP, mp.MP, d)
				break
			}
		}
		for i := 0; i < len(mp.Idx); i++ {
			if math.Abs(float64(mp.Idx[i]-d.expectedMPIdx[i])) > 1e-7 {
				t.Errorf("Expected %d,\nbut got\n%v for\n%+v", d.expectedMPIdx, mp.Idx, d)
				break
			}
		}
	}
}

func TestComputeMpx(t *testing.T) {
	var err error
	var mp *MatrixProfile

	testdata := []struct {
		q             []float64
		t             []float64
		m             int
		p             int
		remap         bool
		expectedMP    []float64
		expectedMPIdx []int
	}{
		{[]float64{}, []float64{}, 2, 1, false, nil, nil},
		{[]float64{1, 1, 1, 1, 1}, []float64{}, 2, 1, false, nil, nil},
		{[]float64{}, []float64{1, 1, 1, 1, 1}, 2, 1, false, nil, nil},
		{[]float64{1, 2, 1, 3, 1}, []float64{2, 1, 1, 2, 1, 3, 1, -1, -2}, 2, 1, false, []float64{0, 0, 0, 0}, []int{2, 3, 2, 3}},
		{[]float64{1, 1, 1, 1, 1}, []float64{1, 1, 1, 1, 1, 2, 2, 3, 4, 5}, 2, 1, false, []float64{2, 2, 2, 2}, []int{0, 1, 2, 3}},
		{[]float64{0, 0.99, 1, 0, 0, 0.98, 1, 0, 0, 0.96, 1, 0}, []float64{0, 0.99, 1, 0, 0, 0.98, 1, 0, 0, 0.96, 1, 0}, 4, 1, false,
			[]float64{0, 0, 0, 0, 0, 0, 0, 0, 0},
			[]int{0, 1, 2, 3, 4, 5, 6, 7, 8}},
		{[]float64{0, 1, 1, 1, 0, 0, 2, 1, 0, 0, 2, 1}, nil, 4, 1, false,
			[]float64{1.9550, 1.8388, 0.8739, 0, 0, 1.9550, 0.8739, 0, 0},
			[]int{4, 2, 6, 7, 8, 1, 2, 3, 4}},
		{[]float64{0, 1, 1, 1, 0, 0, 2, 1, 0, 0, 2, 1}, nil, 4, 1, true,
			[]float64{1.0183, 1.0183, 0.8739, 0, 0, 1.2060, 0.8739, 0, 0},
			[]int{6, 3, 4, 7, 8, 3, 2, 3, 4}},
		{[]float64{0, 0.99, 1, 0, 0, 0.98, 1, 0, 0, 0.96, 1, 0}, nil, 4, 1, false,
			[]float64{0.01435, 0.01435, 0.02913, 0.02913, 0.01435, 0.01435, 0.02913, 0.02913, 0.02913},
			[]int{4, 5, 6, 7, 0, 1, 2, 3, 4}},
		{[]float64{0, 0.99, 1, 0, 0, 0.98, 1, 0, 0, 0.96, 1, 0}, nil, 4, 2, false,
			[]float64{0.01435, 0.01435, 0.02913, 0.02913, 0.01435, 0.01435, 0.02913, 0.02913, 0.02913},
			[]int{4, 5, 6, 7, 0, 1, 2, 3, 4}},
		{[]float64{0, 0.99, 1, 0, 0, 0.98, 1, 0, 0, 0.96, 1, 0}, nil, 4, 4, false,
			[]float64{0.01435, 0.01435, 0.02913, 0.02913, 0.01435, 0.01435, 0.02913, 0.02913, 0.02913},
			[]int{4, 5, 6, 7, 0, 1, 2, 3, 4}},
		{[]float64{0, 0.99, 1, 0, 0, 0.98, 1, 0, 0, 0.96, 1, 0}, nil, 4, 100, false,
			[]float64{0.01435, 0.01435, 0.02913, 0.02913, 0.01435, 0.01435, 0.02913, 0.02913, 0.02913},
			[]int{4, 5, 6, 7, 0, 1, 2, 3, 4}},
	}

	for _, d := range testdata {
		mp, err = New(d.q, d.t, d.m)
		if err != nil {
			if d.expectedMP == nil {
				// Got an error while creating a new matrix profile
				continue
			} else {
				t.Errorf("Did not expect an error, %v,  while creating new mp for %v", err, d)
				return
			}
		}

		o := NewMPOpts()
		o.Algorithm = AlgoMPX
		o.Parallelism = d.p
		o.RemapNegCorr = d.remap
		err = mp.Compute(o)
		if err != nil {
			if d.expectedMP == nil {
				// Got an error while z normalizing and expected an error
				continue
			} else {
				t.Errorf("Did not expect an error, %v, while calculating for %v", err, d)
				break
			}
		}
		if d.expectedMP == nil {
			t.Errorf("Expected an invalid calculation, %+v", d)
			break
		}

		if len(mp.MP) != len(d.expectedMP) {
			t.Errorf("Expected %d elements, but got %d, %+v", len(d.expectedMP), len(mp.MP), d)
			return
		}
		for i := 0; i < len(mp.MP); i++ {
			if math.Abs(mp.MP[i]-d.expectedMP[i]) > 1e-4 {
				t.Errorf("Expected\n%.4f, but got\n%.4f for\n%+v", d.expectedMP, mp.MP, d)
				break
			}
		}
		for i := 0; i < len(mp.Idx); i++ {
			if math.Abs(float64(mp.Idx[i]-d.expectedMPIdx[i])) > 1e-4 {
				t.Errorf("Expected %d,\nbut got\n%v for\n%+v", d.expectedMPIdx, mp.Idx, d)
				break
			}
		}
	}
}

func TestUpdate(t *testing.T) {
	var err error
	var outMP []float64
	var outIdx []int
	var mp *MatrixProfile

	a := []float64{0, 0.99, 1, 0, 0, 0.98, 1, 0, 0, 0.96, 1, 0}

	testdata := []struct {
		vals []float64
	}{
		{[]float64{}},
		{[]float64{0.5}},
		{[]float64{0.2, 0.3, 0.4, 0.9}},
	}

	mp, err = New(a, nil, 4)
	if err != nil {
		t.Error(err)
		return
	}
	o := NewMPOpts()
	o.Algorithm = AlgoSTOMP
	if err = mp.Compute(o); err != nil {
		t.Error(err)
		return
	}

	for _, d := range testdata {
		if err = mp.Update(d.vals); err != nil {
			t.Error(err)
			return
		}
		outMP = make([]float64, len(mp.MP))
		outIdx = make([]int, len(mp.Idx))
		copy(outMP, mp.MP)
		copy(outIdx, mp.Idx)

		if err = mp.stomp(); err != nil {
			t.Error(err)
			return
		}

		for i := 0; i < len(mp.MP); i++ {
			if math.Abs(mp.MP[i]-outMP[i]) > 1e-7 {
				t.Errorf("Expected\n%.4f, but got\n%.4f for\n%+v", mp.MP, outMP, d)
				break
			}
		}
		for i := 0; i < len(mp.Idx); i++ {
			if math.Abs(float64(mp.Idx[i]-outIdx[i])) > 1e-7 {
				t.Errorf("Expected %d,\nbut got\n%v for\n%+v", mp.Idx, outIdx, d)
				break
			}
		}
	}
}

func TestDiscoverDiscords(t *testing.T) {
	mprof := []float64{1, 2, 3, 4}
	a := []float64{1, 2, 3, 4, 5, 6}
	w := 3

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
		mp := MatrixProfile{A: a, B: a, W: w, MP: d.mp, AV: av.Default}
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

		o := NewMPOpts()
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
