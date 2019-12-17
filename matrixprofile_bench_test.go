package matrixprofile

import (
	"math/rand"
	"testing"

	"github.com/matrix-profile-foundation/go-matrixprofile/method"
	"github.com/matrix-profile-foundation/go-matrixprofile/siggen"
	"gonum.org/v1/gonum/fourier"
)

func setupData(numPoints int) []float64 {
	line := siggen.Line(0, 0, numPoints)
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
		qnorm, err = ZNormalize(q)
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
		mean, std, err = movmeanstd(sig, 32)
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

	o := NewOptions()
	o.Method = method.STMP

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

	o := NewOptions()
	o.Method = method.STAMP
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
		reps        int
	}{
		{"m32_p1_pts1024", 32, 1, 1024, 50},
		{"m128_p1_pts1024", 128, 1, 1024, 50},
		{"m128_p2_pts1024", 128, 2, 1024, 100},
		{"m128_p2_pts2048", 128, 2, 2048, 20},
		{"m128_p2_pts4096", 128, 2, 4096, 10},
		{"m128_p2_pts8192", 128, 2, 8192, 10},
	}

	o := NewOptions()

	for _, bm := range benchmarks {
		b.Run(bm.name, func(b *testing.B) {
			sig := setupData(bm.numPoints)
			mp, err := New(sig, nil, 32)
			if err != nil {
				b.Error(err)
			}

			b.N = bm.reps
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

	err = mp.Compute(NewOptions())
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
