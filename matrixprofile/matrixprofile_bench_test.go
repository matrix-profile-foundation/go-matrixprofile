package matrixprofile

import (
	"testing"
)

func setupData() []float64 {
	line := generateLine(0, 0, 120)
	ext := generateLine(0, 100, len(line)/2)
	ext2 := generateLine(0, 600, len(line)/2)
	sig := append(line, ext...)
	sig = append(sig, ext2...)
	noise := generateNoise(10, len(sig))
	sig = sigAdd(sig, noise)

	return sig
}

func BenchmarkSlidingDotProduct(b *testing.B) {
	b.ReportAllocs()
	sig := setupData()
	q := sig[:32]
	var err error
	var cc []float64
	for i := 0; i < b.N; i++ {
		cc, err = slidingDotProduct(q, sig)
		if err != nil {
			b.Error(err)
		}
		if len(cc) < 1 {
			b.Error("expected at least one value from sliding dot product of a timeseries")
		}
	}
}

func BenchmarkMovstd(b *testing.B) {
	b.ReportAllocs()
	sig := setupData()
	var err error
	var std []float64
	for i := 0; i < b.N; i++ {
		std, err = movstd(sig, 32)
		if err != nil {
			b.Error(err)
		}
		if len(std) < 1 {
			b.Error("expected at least one value from moving standard deviation of a timeseries")
		}
	}
}

func BenchmarkZNormalize(b *testing.B) {
	b.ReportAllocs()
	sig := setupData()
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

func BenchmarkMass(b *testing.B) {
	b.ReportAllocs()
	sig := setupData()
	var mp []float64
	var err error
	var q []float64
	for i := 0; i < b.N; i++ {
		q = sig[:32]
		mp, err = Mass(q, sig)
		if err != nil {
			b.Error(err)
		}
		if len(mp) < 1 {
			b.Error("expected at least one value from matrix profile and matrix profile index")
		}
	}
}

func BenchmarkDistanceProfile(b *testing.B) {
	b.ReportAllocs()
	sig := setupData()
	var mp []float64
	var err error
	for i := 0; i < b.N; i++ {
		mp, err = distanceProfile(sig, nil, 32, 0)
		if err != nil {
			b.Error(err)
		}
		if len(mp) < 1 {
			b.Error("expected at least one value from matrix profile and matrix profile index")
		}
	}
}

func BenchmarkStmp(b *testing.B) {
	b.ReportAllocs()
	sig := setupData()
	var mp []float64
	var mpIdx []int
	var err error
	for i := 0; i < b.N; i++ {
		mp, mpIdx, err = Stmp(sig, nil, 32)
		if err != nil {
			b.Error(err)
		}
		if len(mp) < 1 || len(mpIdx) < 1 {
			b.Error("expected at least one value from matrix profile and matrix profile index")
		}
	}
}
