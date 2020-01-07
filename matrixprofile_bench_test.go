package matrixprofile

import (
	"math/rand"
	"testing"

	"github.com/matrix-profile-foundation/go-matrixprofile/siggen"
	"github.com/matrix-profile-foundation/go-matrixprofile/util"
	"gonum.org/v1/gonum/fourier"
)

func setupData(numPoints int) []float64 {
	line := siggen.Line(0, 0, numPoints/2)
	ext := siggen.Line(0, 100, len(line)/2)
	ext2 := siggen.Line(0, 600, len(line)/2)
	sig := siggen.Append(line, ext, ext2)

	noise := siggen.Noise(10, len(sig))
	sig = siggen.Add(sig, noise)

	return sig
}

func BenchmarkZNormalize(b *testing.B) {
	sig := setupData(1000)
	q := sig[:32]
	var err error
	var qnorm []float64
	for i := 0; i < b.N; i++ {
		qnorm, err = util.ZNormalize(q)
		if err != nil {
			b.Error(err)
		}
		if len(qnorm) < 1 {
			b.Error("expected at least one value from z-normalizing a timeseries")
		}
	}
}

func BenchmarkMovmeanstd(b *testing.B) {
	sig := setupData(1000)
	var err error
	var mean, std []float64
	for i := 0; i < b.N; i++ {
		mean, std, err = util.MovMeanStd(sig, 32)
		if err != nil {
			b.Error(err)
		}
		if len(std) < 1 {
			b.Error("expected at least one value from moving standard deviation of a timeseries")
		}
		if len(mean) < 1 {
			b.Error("expected at least one value from moving mean of a timeseries")
		}

	}
}

func BenchmarkCrossCorrelate(b *testing.B) {
	sig := setupData(1000)
	q := sig[:32]
	var err error
	var cc []float64

	mp, err := New(q, sig, 32)
	if err != nil {
		b.Error(err)
	}

	if err = mp.initCaches(); err != nil {
		b.Error(err)
	}

	fft := fourier.NewFFT(mp.N)
	for i := 0; i < b.N; i++ {
		cc = mp.crossCorrelate(q, fft)
		if len(cc) < 1 {
			b.Error("expected at least one value from cross correlation of a timeseries")
		}
	}
}

func BenchmarkMass(b *testing.B) {
	sig := setupData(1000)
	var err error
	var q []float64

	mp, err := New(sig, sig, 32)
	if err != nil {
		b.Error(err)
	}

	if err = mp.initCaches(); err != nil {
		b.Error(err)
	}

	mprof := make([]float64, mp.N-mp.M+1)
	fft := fourier.NewFFT(mp.N)
	for i := 0; i < b.N; i++ {
		q = sig[:32]
		err = mp.mass(q, mprof, fft)
		if err != nil {
			b.Error(err)
		}
		if len(mprof) < 1 {
			b.Error("expected at least one value from matrix profile")
		}
	}
}

func BenchmarkDistanceProfile(b *testing.B) {
	sig := setupData(1000)
	var err error

	mp, err := New(sig, nil, 32)
	if err != nil {
		b.Error(err)
	}

	if err = mp.initCaches(); err != nil {
		b.Error(err)
	}

	mprof := make([]float64, mp.N-mp.M+1)
	fft := fourier.NewFFT(mp.N)
	for i := 0; i < b.N; i++ {
		err = mp.distanceProfile(0, mprof, fft)
		if err != nil {
			b.Error(err)
		}
		if len(mprof) < 1 {
			b.Error("expected at least one value from matrix profile")
		}
	}
}

func BenchmarkCalculateDistanceProfile(b *testing.B) {
	sig := setupData(1000)
	var err error

	mp, err := New(sig, nil, 32)
	if err != nil {
		b.Error(err)
	}

	if err = mp.initCaches(); err != nil {
		b.Error(err)
	}

	fft := fourier.NewFFT(mp.N)
	dot := mp.crossCorrelate(mp.A[:mp.M], fft)

	mprof := make([]float64, len(dot))

	for i := 0; i < b.N; i++ {
		err = mp.calculateDistanceProfile(dot, 0, mprof)
		if err != nil {
			b.Error(err)
		}
		if len(mprof) < 1 {
			b.Error("expected at least one value from matrix profile")
		}
	}
}

func BenchmarkStmp(b *testing.B) {
	sig := setupData(1000)

	benchmarks := []struct {
		name string
		m    int
	}{
		{"m32_pts1k", 32},
		{"m128_pts1k", 128},
	}

	o := NewMPOpts()
	o.Algorithm = AlgoSTMP

	for _, bm := range benchmarks {
		b.Run(bm.name, func(b *testing.B) {
			mp, err := New(sig, nil, bm.m)
			if err != nil {
				b.Error(err)
			}

			for i := 0; i < b.N; i++ {
				err = mp.Compute(o)
				if err != nil {
					b.Error(err)
				}
				if len(mp.MP) < 1 || len(mp.Idx) < 1 {
					b.Error("expected at least one value from matrix profile and matrix profile index")
				}
			}
		})
	}
}

func BenchmarkStamp(b *testing.B) {
	sig := setupData(1000)

	mp, err := New(sig, nil, 32)
	if err != nil {
		b.Error(err)
	}

	o := NewMPOpts()
	o.Algorithm = AlgoSTAMP
	o.Sample = 1.0
	o.Parallelism = 2

	b.Run("m32_p2_pts1k", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			err = mp.Compute(o)
			if err != nil {
				b.Error(err)
			}
			if len(mp.MP) < 1 || len(mp.Idx) < 1 {
				b.Error("expected at least one value from matrix profile and matrix profile index")
			}
		}
	})
}

func BenchmarkStomp(b *testing.B) {
	benchmarks := []struct {
		name        string
		m           int
		parallelism int
		numPoints   int
	}{
		{"m128_p1_pts__1024", 128, 1, 1024},
		{"m128_p2_pts__4096", 128, 2, 4096},
		{"m128_p2_pts_16384", 128, 2, 16384},
		{"m128_p4_pts_16384", 128, 4, 16384},
		{"m1024_p2_pts_16384", 1024, 2, 16384},
	}

	o := NewMPOpts()
	o.Algorithm = AlgoSTOMP

	for _, bm := range benchmarks {
		b.Run(bm.name, func(b *testing.B) {
			sig := setupData(bm.numPoints)
			mp, err := New(sig, nil, bm.m)
			if err != nil {
				b.Error(err)
			}

			o.Parallelism = bm.parallelism
			for i := 0; i < b.N; i++ {
				err = mp.Compute(o)
				if err != nil {
					b.Error(err)
				}
				if len(mp.MP) < 1 || len(mp.Idx) < 1 {
					b.Error("expected at least one value from matrix profile and matrix profile index")
				}
			}
		})
	}
}

func BenchmarkMpx(b *testing.B) {
	benchmarks := []struct {
		name        string
		m           int
		parallelism int
		numPoints   int
	}{
		{"m128_p1_pts__1024", 128, 1, 1024},
		{"m128_p2_pts__4096", 128, 2, 4096},
		{"m128_p2_pts_16384", 128, 2, 16384},
		{"m128_p4_pts_16384", 128, 4, 16384},
		{"m1024_p2_pts_16384", 1024, 2, 16384},
	}

	o := NewMPOpts()
	o.Algorithm = AlgoMPX

	for _, bm := range benchmarks {
		b.Run(bm.name, func(b *testing.B) {
			sig := setupData(bm.numPoints)
			mp, err := New(sig, nil, bm.m)
			if err != nil {
				b.Error(err)
			}

			o.Parallelism = bm.parallelism
			for i := 0; i < b.N; i++ {
				err = mp.Compute(o)
				if err != nil {
					b.Error(err)
				}
				if len(mp.MP) < 1 || len(mp.Idx) < 1 {
					b.Error("expected at least one value from matrix profile and matrix profile index")
				}
			}
		})
	}
}

func BenchmarkUpdate(b *testing.B) {
	sig := setupData(5000)
	mp, err := New(sig, nil, 32)
	if err != nil {
		b.Error(err)
	}

	err = mp.Compute(NewMPOpts())
	if err != nil {
		b.Error(err)
	}

	if len(mp.MP) < 1 || len(mp.Idx) < 1 {
		b.Error("expected at least one value from matrix profile and matrix profile index")
	}

	for i := 0; i < b.N; i++ {
		err = mp.Update([]float64{rand.Float64() - 0.5})
	}
}
