package matrixprofile

import (
	"math"
	"testing"
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
		out, err = zNormalize(d.data)
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
		{[]float64{1, 1, 1, 1}, 2, []float64{1, 1, 1}, []float64{0, 0, 0}},
		{[]float64{-1, -1, -1, -1}, 2, []float64{-1, -1, -1}, []float64{0, 0, 0}},
		{[]float64{1, -1, -1, 1}, 2, []float64{0, -1, 0}, []float64{1, 0, 1}},
		{[]float64{1, -1, -1, 1}, 4, nil, nil},
		{[]float64{1, 2, 4, 8}, 2, []float64{1.5, 3, 6}, []float64{0.5, 1, 2}},
	}

	for _, d := range testdata {
		mean, std, err = movmeanstd(d.data, d.m)
		if err != nil && d.expectedStd == nil && d.expectedMean == nil {
			// Got an error while calculating and expected an error
			continue
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

func TestCrossCorrelate(t *testing.T) {
	var err error
	var out []float64
	var mp *MatrixProfile

	testdata := []struct {
		q        []float64
		t        []float64
		expected []float64
	}{
		{[]float64{}, []float64{}, nil},
		{[]float64{1, 1, 1, 1, 1}, []float64{}, nil},
		{[]float64{}, []float64{1, 1, 1, 1, 1}, nil},
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
		out, err = mp.crossCorrelate(d.q)
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
		err = mp.mass(d.q, out)
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
		err = mp.distanceProfile(d.idx, mprof)
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

		dot, err := mp.crossCorrelate(mp.a[:mp.m])
		if err != nil {
			t.Error(err)
		}

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

		err = mp.Stamp(d.sample)
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

func TestStomp(t *testing.T) {
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

		err = mp.Stomp()
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

func TestDiscords(t *testing.T) {
	a := []float64{0, 0, 0.50, 0.99, 0.99, 0.50, 0, 0, 0, 0.50, 0.99, 0.10, 0.50, 0, 0, 0, 0.50, 0.99, 0.99, 0.50, 0, 0, 0}
	a = SigAdd(a, Noise(1e-7, len(a)))

	testdata := []struct {
		a                []float64
		b                []float64
		k                int
		expectedDiscords []int
	}{
		{
			a, nil, 3,
			[]int{9, 11, 19},
		},
	}

	for _, d := range testdata {
		mp, err := New(d.a, d.b, 4)
		if err != nil {
			t.Error(err)
		}
		if err = mp.Stmp(); err != nil {
			t.Error(err)
		}
		discords := mp.Discords(d.k)
		if err != nil {
			if d.expectedDiscords == nil {
				continue
			}
			t.Error(err)
		}

		for i, idx := range discords {
			if idx != d.expectedDiscords[i] {
				t.Errorf("expected index, %d, but got %d", d.expectedDiscords[i], idx)
			}
		}
	}
}

func TestTopKMotifs(t *testing.T) {
	a := []float64{0, 0, 0.56, 0.99, 0.97, 0.75, 0, 0, 0, 0.43, 0.98, 0.99, 0.65, 0, 0, 0, 0.6, 0.97, 0.965, 0.8, 0, 0, 0}
	a = SigAdd(a, Noise(1e-7, len(a)))

	testdata := []struct {
		a               []float64
		b               []float64
		k               int
		expectedMotifs  [][]int
		expectedMinDist []float64
	}{
		{
			a, nil, 3,
			[][]int{{1, 15}, {0, 7, 14}, {3, 10}},
			[]float64{0.1459618197766371, 0.3352336136782056, 0.46369664551715467},
		},
		{
			a, a, 3,
			nil,
			nil,
		},
		{
			a, nil, 5,
			[][]int{{1, 15}, {0, 7, 14}, {3, 10}, {}, {}},
			[]float64{0.1459618197766371, 0.3352336136782056, 0.46369664551715467, 0, 0},
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
				t.Errorf("expected %d motifs for group %d, but got %d", len(d.expectedMotifs[i]), i, len(mg.Idx))
			}

			for j, idx := range mg.Idx {
				if idx != d.expectedMotifs[i][j] {
					t.Errorf("expected index, %d for group %d, but got %d", d.expectedMotifs[i][j], i, idx)
				}
			}
			if math.Abs(mg.MinDist-d.expectedMinDist[i]) > 1e-7 {
				t.Errorf("expected minimum distance, %.3f for group %d, but got %.3f", d.expectedMinDist[i], i, mg.MinDist)
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
		mp := MatrixProfile{Idx: d.mpIdx}
		minIdx, minVal, histo = mp.Segment()
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
