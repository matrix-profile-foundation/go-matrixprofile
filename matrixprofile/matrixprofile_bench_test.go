package matrixprofile

import (
	"math/rand"
	"testing"
)

func setupData(numPoints int) []float64 {
	line := Line(0, 0, numPoints)
	ext := Line(0, 100, len(line)/2)
	ext2 := Line(0, 600, len(line)/2)
	sig := append(line, ext...)
	sig = append(sig, ext2...)
	noise := Noise(10, len(sig))
	sig = SigAdd(sig, noise)

	return sig
}

func BenchmarkZNormalize(b *testing.B) {
	sig := setupData(1000)
	q := sig[:32]
	var err error
	var qnorm []float64
	for i := 0; i < b.N; i++ {
		qnorm, err = zNormalize(q)
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

	for i := 0; i < b.N; i++ {
		cc, err = mp.crossCorrelate(q)
		if err != nil {
			b.Error(err)
		}
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

	mprof := make([]float64, mp.n-mp.m+1)
	for i := 0; i < b.N; i++ {
		q = sig[:32]
		err = mp.mass(q, mprof)
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

	mprof := make([]float64, mp.n-mp.m+1)
	for i := 0; i < b.N; i++ {
		err = mp.distanceProfile(0, mprof)
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

	dot, err := mp.crossCorrelate(mp.a[:mp.m])
	if err != nil {
		b.Error(err)
	}

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
		{"m16", 16},
		{"m32", 32},
		{"m64", 64},
		{"m128", 128},
	}

	for _, bm := range benchmarks {
		b.Run(bm.name, func(b *testing.B) {
			mp, err := New(sig, nil, 32)
			if err != nil {
				b.Error(err)
			}

			for i := 0; i < b.N; i++ {
				err = mp.Stmp()
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

func BenchmarkStomp(b *testing.B) {
	benchmarks := []struct {
		name        string
		m           int
		parallelism int
		numPoints   int
		reps        int
	}{
		{"m16_p1_pts1k", 16, 1, 1000, 50},
		{"m128_p1_pts1k", 128, 1, 1000, 50},
		{"m128_p2_pts1k", 128, 2, 1000, 100},
		{"m128_p2_pts2k", 128, 2, 2000, 20},
		{"m128_p2_pts5k", 128, 2, 5000, 10},
	}

	for _, bm := range benchmarks {
		b.Run(bm.name, func(b *testing.B) {
			sig := setupData(bm.numPoints)
			mp, err := New(sig, nil, 32)
			if err != nil {
				b.Error(err)
			}

			b.N = bm.reps
			for i := 0; i < b.N; i++ {
				err = mp.Stomp(bm.parallelism)
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

func BenchmarkStompUpdate(b *testing.B) {
	sig := setupData(5000)
	mp, err := New(sig, nil, 32)
	if err != nil {
		b.Error(err)
	}

	err = mp.Stomp(2)
	if err != nil {
		b.Error(err)
	}

	if len(mp.MP) < 1 || len(mp.Idx) < 1 {
		b.Error("expected at least one value from matrix profile and matrix profile index")
	}

	for i := 0; i < b.N; i++ {
		err = mp.StampUpdate([]float64{rand.Float64() - 0.5})
	}
}
