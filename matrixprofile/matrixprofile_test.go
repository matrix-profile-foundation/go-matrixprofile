package matrixprofile

import (
	"math"
	"testing"

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
		{[]float64{}, []float64{1, 1, 1, 1, 1}, 2, true},
		{[]float64{1, 2, 3, 4, 5}, []float64{1, 1, 1, 1, 1}, 2, false},
		{[]float64{1, 2, 3, 4, 5}, []float64{1, 1, 1, 1, 1}, 1, true},
		{[]float64{1, 2, 3, 4, 5}, []float64{1, 1, 1, 1, 1}, 4, true},
	}

	for _, d := range testdata {
		_, err := New(d.a, d.b, d.m)
		if d.expectedErr && err == nil {
			t.Errorf("Expected an error, but got none for %v", d)
		}
		if !d.expectedErr && err != nil {
			t.Errorf("Expected no error, but got %v for %v", err, d)
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
		if err != nil && d.expected == nil {
			// Got an error while creating a new matrix profile
			continue
		}

		fft := fourier.NewFFT(mp.n)
		out = mp.crossCorrelate(d.q, fft)
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
		out = make([]float64, mp.n-mp.m+1)
		fft := fourier.NewFFT(mp.n)
		err = mp.mass(d.q, out, fft)
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

		mprof = make([]float64, mp.n-mp.m+1)
		fft := fourier.NewFFT(mp.n)
		err = mp.distanceProfile(d.idx, mprof, fft)
		if err != nil && d.expectedMP == nil {
			// Got an error while z normalizing and expected an error
			continue
		}
		if d.expectedMP == nil {
			t.Errorf("Expected an invalid distance profile calculation, %+v", d)
		}
		if err != nil {
			t.Errorf("Did not expect error, %v\n%+v", err, d)
		}
		if len(mprof) != len(d.expectedMP) {
			t.Errorf("Expected %d elements, but got %d\n%+v", len(d.expectedMP), len(mprof), d)
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

		fft := fourier.NewFFT(mp.n)
		dot := mp.crossCorrelate(mp.a[:mp.m], fft)

		mprof = make([]float64, mp.n-mp.m+1)
		err = mp.calculateDistanceProfile(dot, d.idx, mprof)
		if err != nil && d.expectedMP == nil {
			// Got an error while z normalizing and expected an error
			continue
		}
		if d.expectedMP == nil {
			t.Errorf("Expected an invalid distance profile calculation, %+v", d)
		}
		if err != nil {
			t.Errorf("Did not expect error, %v\n%+v", err, d)
		}
		if len(mprof) != len(d.expectedMP) {
			t.Errorf("Expected %d elements, but got %d\n%+v", len(d.expectedMP), len(mprof), d)
		}
		for i := 0; i < len(mprof); i++ {
			if math.Abs(mprof[i]-d.expectedMP[i]) > 1e-7 {
				t.Errorf("Expected\n%.7f, but got\n%.7f for\n%+v", d.expectedMP, mprof, d)
				break
			}
		}
	}

}

func TestStmp(t *testing.T) {
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

		err = mp.Stmp()
		if err != nil && d.expectedMP == nil {
			// Got an error while z normalizing and expected an error
			continue
		}
		if d.expectedMP == nil {
			t.Errorf("Expected an invalid STMP calculation, %+v", d)
		}
		if err != nil {
			t.Errorf("Did not expect error, %v, %+v", err, d)
		}
		if len(mp.MP) != len(d.expectedMP) {
			t.Errorf("Expected %d elements, but got %d, %+v", len(d.expectedMP), len(mp.MP), d)
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

func TestStamp(t *testing.T) {
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

		err = mp.Stamp(d.sample, 2)
		if err != nil && d.expectedMP == nil {
			// Got an error while z normalizing and expected an error
			continue
		}
		if d.expectedMP == nil {
			t.Errorf("Expected an invalid STAMP calculation, %+v", d)
		}
		if err != nil {
			t.Errorf("Did not expect error, %v, %+v", err, d)
		}
		if len(mp.MP) != len(d.expectedMP) {
			t.Errorf("Expected %d elements, but got %d, %+v", len(d.expectedMP), len(mp.MP), d)
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

func TestStomp(t *testing.T) {
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
		{[]float64{1, 1}, []float64{1, 1, 1, 1, 1}, 2, 1, []float64{math.Inf(1), math.Inf(1), math.Inf(1), math.Inf(1)}, []int{0, math.MaxInt64, math.MaxInt64, math.MaxInt64}},
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
		if err != nil && d.expectedMP == nil {
			// Got an error while creating a new matrix profile
			continue
		}

		err = mp.Stomp(d.p)
		if err != nil && d.expectedMP == nil {
			// Got an error while z normalizing and expected an error
			continue
		}
		if d.expectedMP == nil {
			t.Errorf("Expected an invalid STOMP calculation, %+v", d)
			break
		}
		if err != nil {
			t.Errorf("Did not expect error, %v, %+v", err, d)
			break
		}
		if len(mp.MP) != len(d.expectedMP) {
			t.Errorf("Expected %d elements, but got %d, %+v", len(d.expectedMP), len(mp.MP), d)
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

func TestStampUpdate(t *testing.T) {
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
	}
	if err = mp.Stomp(1); err != nil {
		t.Error(err)
	}

	for _, d := range testdata {
		if err = mp.StampUpdate(d.vals); err != nil {
			t.Error(err)
		}
		outMP = make([]float64, len(mp.MP))
		outIdx = make([]int, len(mp.Idx))
		copy(outMP, mp.MP)
		copy(outIdx, mp.Idx)

		if err = mp.Stomp(1); err != nil {
			t.Error(err)
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

func TestDiscords(t *testing.T) {
	mprof := []float64{1, 2, 3, 4}

	testdata := []struct {
		mp               []float64
		k                int
		exzone           int
		expectedDiscords []int
	}{
		{mprof, 4, 0, []int{3, 3, 3, 3}},
		{mprof, 4, 1, []int{3, 1, math.MaxInt64, math.MaxInt64}},
		{mprof, 10, 1, []int{3, 1, math.MaxInt64, math.MaxInt64}},
		{mprof, 0, 1, []int{}},
		{[]float64{}, 3, 1, []int{}},
	}

	for _, d := range testdata {
		mp := MatrixProfile{MP: d.mp}
		discords := mp.Discords(d.k, d.exzone)
		if len(discords) != len(d.expectedDiscords) {
			t.Errorf("Got a length of %d discords, but expected %d, for %v", len(discords), len(d.expectedDiscords), d)
		}
		for i, idx := range discords {
			if idx != d.expectedDiscords[i] {
				t.Errorf("expected index, %d, but got %d, for %v", d.expectedDiscords[i], idx, d)
			}
		}
	}
}

func TestTopKMotifs(t *testing.T) {
	a := []float64{0, 0, 0.56, 0.99, 0.97, 0.75, 0, 0, 0, 0.43, 0.98, 0.99, 0.65, 0, 0, 0, 0.6, 0.97, 0.965, 0.8, 0, 0, 0}

	testdata := []struct {
		a               []float64
		b               []float64
		k               int
		expectedMotifs  [][]int
		expectedMinDist []float64
	}{
		{
			a, nil, 3,
			[][]int{{0, 14}, {0, 7, 14}, {3, 10}},
			[]float64{0.1459619228330262, 0.3352336136782056, 0.46369664551715467},
		},
		{
			a, a, 3,
			nil,
			nil,
		},
		{
			a, nil, 5,
			[][]int{{0, 14}, {0, 7, 14}, {3, 10}, {}, {}},
			[]float64{0.1459619228330262, 0.3352336136782056, 0.46369664551715467, 0, 0},
		},
	}

	for _, d := range testdata {
		mp, err := New(d.a, d.b, 7)
		if err != nil {
			t.Error(err)
		}
		if err = mp.Stmp(); err != nil {
			t.Error(err)
		}
		motifs, err := mp.TopKMotifs(d.k, 2)
		if err != nil {
			if d.expectedMotifs == nil {
				continue
			}
			t.Error(err)
		}

		for i, mg := range motifs {
			if len(mg.Idx) != len(d.expectedMotifs[i]) {
				t.Errorf("expected %d motifs for group %d, but got %d for %v", len(d.expectedMotifs[i]), i, len(mg.Idx), d)
			}

			for j, idx := range mg.Idx {
				if idx != d.expectedMotifs[i][j] {
					t.Errorf("expected index, %d for group %d, but got %d for %v", d.expectedMotifs[i][j], i, idx, d)
				}
			}
			if math.Abs(mg.MinDist-d.expectedMinDist[i]) > 1e-7 {
				t.Errorf("expected minimum distance, %v for group %d, but got %v for %v", d.expectedMinDist[i], i, mg.MinDist, d)
			}
		}
	}
}

func TestApplyAV(t *testing.T) {
	mprof := []float64{4, 6, 10, 2, 1, 0, 1, 2, 0, 0, 1, 2, 6}

	testdata := []struct {
		av         []float64
		expectedMP []float64
	}{
		{[]float64{}, nil},
		{[]float64{1, 1, 1, 1, 1}, nil},
		{[]float64{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, mprof},
		{[]float64{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, []float64{14, 16, 20, 12, 11, 10, 11, 12, 10, 10, 11, 12, 16}},
		{[]float64{1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1}, []float64{4, 6, 10, 2, 1, 0, 1, 2, 10, 10, 1, 2, 6}},
		{[]float64{1, 1, 1, 1, 1, 1, 1.01, 1, 0, 0, 1, 1, 1}, nil},
		{[]float64{1, 1, 1, 1, 1, 1, 1, 1, -0.01, 0, 1, 1, 1}, nil},
	}

	var mp MatrixProfile
	var err error
	var out []float64
	for _, d := range testdata {
		newMP := make([]float64, len(mprof))
		copy(newMP, mprof)
		mp = MatrixProfile{MP: newMP}
		out, err = mp.ApplyAV(d.av)
		if err != nil && d.expectedMP == nil {
			// Expected error while applying av
			continue
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
