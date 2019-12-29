package matrixprofile

import (
	"math"
	"testing"

	"gonum.org/v1/gonum/fourier"
)

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
		out = make([]float64, mp.N-mp.M+1)
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

		mprof = make([]float64, mp.N-mp.M+1)
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
		dot := mp.crossCorrelate(mp.A[:mp.M], fft)

		mprof = make([]float64, mp.N-mp.M+1)
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

		o := NewComputeOpts()
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

		o := NewComputeOpts()
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

		o := NewComputeOpts()
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
		expectedMP    []float64
		expectedMPIdx []int
	}{
		{[]float64{}, []float64{}, 2, 1, nil, nil},
		{[]float64{1, 1, 1, 1, 1}, []float64{}, 2, 1, nil, nil},
		{[]float64{}, []float64{1, 1, 1, 1, 1}, 2, 1, nil, nil},
		{[]float64{1, 2, 1, 3, 1}, []float64{2, 1, 1, 2, 1, 3, 1, -1, -2}, 2, 1, []float64{0, 0, 0, 0}, []int{2, 3, 2, 3}},
		{[]float64{1, 1, 1, 1, 1}, []float64{1, 1, 1, 1, 1, 2, 2, 3, 4, 5}, 2, 1, []float64{2, 2, 2, 2}, []int{0, 1, 2, 3}},
		{[]float64{0, 0.99, 1, 0, 0, 0.98, 1, 0, 0, 0.96, 1, 0}, []float64{0, 0.99, 1, 0, 0, 0.98, 1, 0, 0, 0.96, 1, 0}, 4, 1,
			[]float64{0, 0, 0, 0, 0, 0, 0, 0, 0},
			[]int{0, 1, 2, 3, 4, 5, 6, 7, 8}},
		{[]float64{0, 1, 1, 1, 0, 0, 2, 1, 0, 0, 2, 1}, nil, 4, 1,
			[]float64{1.9550, 1.8388, 0.8739, 0, 0, 1.9550, 0.8739, 0, 0},
			[]int{4, 2, 6, 7, 8, 1, 2, 3, 4}},
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

		o := NewComputeOpts()
		o.Algorithm = AlgoMPX
		o.Parallelism = d.p
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
			if math.Abs(float64(mp.Idx[i]-d.expectedMPIdx[i])) > 1e-7 {
				t.Errorf("Expected %d,\nbut got\n%v for\n%+v", d.expectedMPIdx, mp.Idx, d)
				break
			}
		}
	}
}

func TestComputePmp(t *testing.T) {
	var err error
	var mp *MatrixProfile

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
		mp, err = New(d.a, d.b, d.lb)
		if err != nil {
			if d.expectedPMP == nil {
				// Got an error while creating a new matrix profile
				continue
			} else {
				t.Errorf("Did not expect an error, %v,  while creating new mp for %v", err, d)
				return
			}
		}

		o := NewComputeOpts()
		o.Algorithm = AlgoPMP
		o.Parallelism = d.p
		o.LowerM = d.lb
		o.UpperM = d.ub
		err = mp.Compute(o)
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

		if len(mp.PMP) != len(d.expectedPMP) {
			t.Errorf("Expected %d elements, but got %d, %+v", len(d.expectedPMP), len(mp.PMP), d)
			return
		}
		for j := 0; j < len(mp.PMP); j++ {
			if len(mp.PMP[j]) != len(d.expectedPMP[j]) {
				t.Errorf("Expected %d elements, but got %d, %+v", len(d.expectedPMP[j]), len(mp.PMP[j]), d)
				return
			}
			for i := 0; i < len(mp.PMP[j]); i++ {
				if math.Abs(mp.PMP[j][i]-d.expectedPMP[j][i]) > 1e-4 {
					t.Errorf("Expected\n%.6f, but got\n%.6f for\n%+v", d.expectedPMP[j], mp.PMP[j], d)
					break
				}
			}
			for i := 0; i < len(mp.PIdx[j]); i++ {
				if math.Abs(float64(mp.PIdx[j][i]-d.expectedPIdx[j][i])) > 1e-7 {
					t.Errorf("Expected %d,\nbut got\n%v for\n%+v", d.expectedPIdx[j], mp.PIdx[j], d)
					break
				}
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
	o := NewComputeOpts()
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

		if err = mp.stomp(1); err != nil {
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
