package siggen

import (
	"testing"
)

func TestSin(t *testing.T) {
	testdata := []struct {
		fs        float64
		duration  float64
		expectedN int
	}{
		{0, 10, 0},
		{100, 1, 100},
		{100, 1.5, 150},
		{100, 0, 0},
	}

	var out []float64
	for _, d := range testdata {
		out = Sin(1, 5, 0, 0, d.fs, d.duration)
		if len(out) != d.expectedN {
			t.Errorf("expected output length, %d, but got, %d, for %v", d.expectedN, len(out), d)
		}
	}
}

func TestSawtooth(t *testing.T) {
	testdata := []struct {
		fs        float64
		duration  float64
		expectedN int
	}{
		{0, 10, 0},
		{100, 1, 100},
		{100, 1.5, 150},
		{100, 0, 0},
	}

	var out []float64
	for _, d := range testdata {
		out = Sawtooth(1, 5, 0, 0, d.fs, d.duration)
		if len(out) != d.expectedN {
			t.Errorf("expected output length, %d, but got, %d, for %v", d.expectedN, len(out), d)
		}
	}

}

func TestLine(t *testing.T) {
	testdata := []struct {
		n         int
		expectedN int
	}{
		{0, 0},
		{100, 100},
	}

	var out []float64
	for _, d := range testdata {
		out = Line(1, 0, d.n)
		if len(out) != d.expectedN {
			t.Errorf("expected output length, %d, but got, %d, for %v", d.expectedN, len(out), d)
		}
	}
}

func TestNoise(t *testing.T) {
	testdata := []struct {
		n         int
		expectedN int
	}{
		{0, 0},
		{100, 100},
	}

	var out []float64
	for _, d := range testdata {
		out = Noise(0, d.n)
		if len(out) != d.expectedN {
			t.Errorf("expected output length, %d, but got, %d, for %v", d.expectedN, len(out), d)
		}
	}
}

func TestAdd(t *testing.T) {
	testdata := []struct {
		sig1        []float64
		sig2        []float64
		expectedOut []float64
	}{
		{Line(0, 2, 100), Line(0, 5, 100), Line(0, 7, 100)},
		{Line(0, 2, 200), Line(0, 5, 100), Append(Line(0, 7, 100), Line(0, 2, 100))},
		{Line(0, 2, 100), Line(0, 5, 200), Append(Line(0, 7, 100), Line(0, 5, 100))},
	}

	var out []float64
	for _, d := range testdata {
		out = Add(d.sig1, d.sig2)
		if len(out) != len(d.expectedOut) {
			t.Errorf("expected output length, %d, but got, %d, for %v", len(d.expectedOut), len(out), d)
			break
		}

		for i, val := range out {
			if val != d.expectedOut[i] {
				t.Errorf("expected value of %.3f at index %d, but got %.3f for %v", d.expectedOut[i], i, val, d)
				break
			}
		}
	}
}

func TestAppend(t *testing.T) {
	testdata := []struct {
		sig1      []float64
		sig2      []float64
		expectedN int
	}{
		{Line(0, 2, 100), Line(0, 5, 100), 200},
		{Line(0, 2, 200), Line(0, 5, 100), 300},
	}

	var out []float64
	for _, d := range testdata {
		out = Append(d.sig1, d.sig2)
		if len(out) != d.expectedN {
			t.Errorf("expected output length, %d, but got, %d, for %v", d.expectedN, len(out), d)
			break
		}
	}
}
