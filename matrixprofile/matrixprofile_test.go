package matrixprofile

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

func TestMovstd(t *testing.T) {
	var err error
	var out []float64

	testdata := []struct {
		data     []float64
		m        int
		expected []float64
	}{
		{[]float64{}, 4, nil},
		{[]float64{}, 0, nil},
		{[]float64{1, 1, 1, 1}, 0, nil},
		{[]float64{1, 1, 1, 1}, 2, []float64{0, 0, 0}},
		{[]float64{-1, -1, -1, -1}, 2, []float64{0, 0, 0}},
		{[]float64{1, -1, -1, 1}, 2, []float64{1, 0, 1}},
		{[]float64{1, -1, -1, 1}, 4, nil},
		{[]float64{1, 2, 4, 8}, 2, []float64{0.5, 1, 2}},
	}

	for _, d := range testdata {
		out, err = movstd(d.data, d.m)
		if err != nil && d.expected == nil {
			// Got an error while calculating and expected an error
			continue
		}
		if d.expected == nil {
			t.Errorf("Expected an invalid moving standard deviation, %v", d)
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

func TestSlidingDotProduct(t *testing.T) {
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
		out, err = mp.slidingDotProduct(d.q)
		if err != nil && d.expected == nil {
			// Got an error while z normalizing and expected an error
			continue
		}
		if d.expected == nil {
			t.Errorf("Expected an invalid sliding dot product calculation, %v", d)
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
		out, err = mp.mass(d.q)
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
	}

	for _, d := range testdata {
		mp, err = New(d.q, d.t, d.m)
		if err != nil && d.expectedMP == nil {
			// Got an error while creating a new matrix profile
			continue
		}

		mprof, err = mp.distanceProfile(d.idx)
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
