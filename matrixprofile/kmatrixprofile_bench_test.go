package matrixprofile

import (
	"testing"

	"github.com/aouyang1/go-matrixprofile/siggen"
)

func setupKData() [][]float64 {
	sin := siggen.Sin(1, 4, 0, 0, 100, 0.25)
	saw := siggen.Sawtooth(1, 4, 0, 0, 100, 0.25)
	square := siggen.Square(1, 4, 0, 0, 100, 0.25)
	line := siggen.Line(0, 0, len(sin)*4)
	line2 := siggen.Line(0, 0, len(sin)*3)
	sig := make([][]float64, 3)
	sig[0] = siggen.Append(line, line, line, saw, line2, saw, line2)
	sig[1] = siggen.Append(line, sin, line2, sin, line2, sin, line2, sin, line2)
	sig[2] = siggen.Append(line, square, line2, square, line2, square, line2, square, line2)

	noise := siggen.Noise(0.1, len(sig[0]))
	sig[0] = siggen.Add(sig[0], noise)

	noise = siggen.Noise(0.1, len(sig[0]))
	sig[1] = siggen.Add(sig[1], noise)

	noise = siggen.Noise(0.1, len(sig[0]))
	sig[2] = siggen.Add(sig[2], noise)

	return sig
}

func BenchmarkMStomp(b *testing.B) {
	sig := setupKData()
	mp, err := NewK(sig, 25)
	if err != nil {
		b.Error(err)
	}

	for i := 0; i < b.N; i++ {
		err = mp.MStomp()
		if err != nil {
			b.Error(err)
		}
		if len(mp.MP) < 1 || len(mp.Idx) < 1 {
			b.Error("expected at least one dimension from matrix profile and matrix profile index")
		}
	}
}
