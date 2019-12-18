package util

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/stat"
)

// ZNormalize computes a z-normalized version of a slice of floats.
// This is represented by y[i] = (x[i] - mean(x))/std(x)
func ZNormalize(ts []float64) ([]float64, error) {
	var i int

	if len(ts) == 0 {
		return nil, fmt.Errorf("slice does not have any data")
	}

	m := stat.Mean(ts, nil)

	out := make([]float64, len(ts))
	for i = 0; i < len(ts); i++ {
		out[i] = ts[i] - m
	}

	var std float64
	for _, val := range out {
		std += val * val
	}
	std = math.Sqrt(std / float64(len(out)))

	if std == 0 {
		return out, fmt.Errorf("standard deviation is zero")
	}

	for i = 0; i < len(ts); i++ {
		out[i] = out[i] / std
	}

	return out, nil
}

// MovMeanStd computes the mean and standard deviation of each sliding
// window of m over a slice of floats. This is done by one pass through
// the data and keeping track of the cumulative sum and cumulative sum
// squared.  s between these at intervals of m provide a total of O(n)
// calculations for the standard deviation of each window of size m for
// the time series ts.
func MovMeanStd(ts []float64, m int) ([]float64, []float64, error) {
	if m <= 1 {
		return nil, nil, fmt.Errorf("length of slice must be greater than 1")
	}

	if m > len(ts) {
		return nil, nil, fmt.Errorf("m cannot be greater than length of slice")
	}

	var i int

	c := make([]float64, len(ts)+1)
	csqr := make([]float64, len(ts)+1)
	for i = 0; i < len(ts)+1; i++ {
		if i == 0 {
			c[i] = 0
			csqr[i] = 0
		} else {
			c[i] = ts[i-1] + c[i-1]
			csqr[i] = ts[i-1]*ts[i-1] + csqr[i-1]
		}
	}

	mean := make([]float64, len(ts)-m+1)
	std := make([]float64, len(ts)-m+1)
	for i = 0; i < len(ts)-m+1; i++ {
		mean[i] = (c[i+m] - c[i]) / float64(m)
		std[i] = math.Sqrt((csqr[i+m]-csqr[i])/float64(m) - mean[i]*mean[i])
	}

	return mean, std, nil
}

// ApplyExclusionZone performs an in place operation on a given matrix
// profile setting distances around an index to +Inf
func ApplyExclusionZone(profile []float64, idx, zoneSize int) {
	startIdx := 0
	if idx-zoneSize > startIdx {
		startIdx = idx - zoneSize
	}
	endIdx := len(profile)
	if idx+zoneSize < endIdx {
		endIdx = idx + zoneSize
	}
	for i := startIdx; i < endIdx; i++ {
		profile[i] = math.Inf(1)
	}
}

// ArcCurve computes the arc curve (histogram) which is uncorrected for.
// This loops through the matrix profile index and increments the
// counter for each index that the destination index passes through
// start from the index in the matrix profile index.
func ArcCurve(mpIdx []int) []float64 {
	histo := make([]float64, len(mpIdx))
	for i, idx := range mpIdx {
		switch {
		case idx >= len(mpIdx):
		case idx < 0:
			continue
		case idx > i+1:
			for j := i + 1; j < idx; j++ {
				histo[j]++
			}
		case idx < i-1:
			for j := i - 1; j > idx; j-- {
				histo[j]++
			}
		}
	}
	return histo
}

// Iac represents the ideal arc curve with a maximum of n/2 and 0 values
// at 0 and n-1. The derived equation to ensure the requirements is
// -(sqrt(2/n)*(x-n/2))^2 + n/2 = y
func Iac(x float64, n int) float64 {
	return -math.Pow(math.Sqrt(2/float64(n))*(x-float64(n)/2.0), 2.0) + float64(n)/2.0
}

func MuInvN(a []float64, w int) ([]float64, []float64) {
	mu := Sum2s(a, w)
	sig := make([]float64, len(a)-w+1)
	h := make([]float64, len(a))
	r := make([]float64, len(a))

	var mu_a, c float64
	var a1, a2, a3, p, s, x, z float64
	for i := 0; i < len(mu); i++ {
		for j := i; j < i+w; j++ {
			mu_a = a[j] - mu[i]
			h[j] = mu_a * mu_a

			c = (math.Pow(2.0, 27.0) + 1) * mu_a
			a1 = c - (c - mu_a)
			a2 = mu_a - a1
			a3 = a1 * a2
			r[j] = a2*a2 - (((h[j] - a1*a1) - a3) - a3)
		}

		p = h[i]
		s = r[i]

		for j := i + 1; j < i+w; j++ {
			x = p + h[j]
			z = x - p
			s += ((p - (x - z)) + (h[j] - z)) + r[j]
			p = x
		}

		sig[i] = 1 / math.Sqrt(p+s)
	}
	return mu, sig
}

func Sq2s(a []float64) float64 {
	c := math.Pow(2.0, 27.0) + 1
	h := make([]float64, len(a))
	r := make([]float64, len(a))

	var a1, a2, a3, p, s float64
	for i := 0; i < len(a); i++ {
		h[i] = a[i] * a[i]

		a1 = c*a[i] - (c*a[i] - a[i])
		a2 = a[i] - a1
		a3 = a1 * a2
		r[i] = a2*a2 - (((h[i] - a1*a1) - a3) - a3)
	}
	p = h[0]
	s = r[0]

	var x, z float64
	for i := 1; i < len(a); i++ {
		x = p + h[i]
		z = x - p
		s += ((p - (x - z)) + (h[i] - z)) + r[i]
		p = x
	}

	return p + s
}

func TwoSquare(a []float64) ([]float64, []float64) {
	c := math.Pow(2.0, 27.0) + 1
	var a1, a2, a3 float64
	y := make([]float64, len(a))
	x := make([]float64, len(a))
	for i := 0; i < len(a); i++ {
		x[i] = a[i] * a[i]

		a1 = c*a[i] - (c*a[i] - a[i])
		a2 = a[i] - a1
		a3 = a1 * a2
		y[i] = a2*a2 - (((x[i] - a1*a1) - a3) - a3)
	}

	return x, y
}

func Sum2s(a []float64, w int) []float64 {
	if len(a) < w {
		return nil
	}
	p := a[0]
	s := 0.0
	var x, z float64
	for i := 1; i < w; i++ {
		x = p + a[i]
		z = x - p
		s += (p - (x - z)) + (a[i] - z)
		p = x
	}

	res := make([]float64, len(a)-w+1)
	res[0] = (p + s) / float64(w)
	for i := w; i < len(a); i++ {
		x = p - a[i-w]
		z = x - p
		s += (p - (x - z)) - (a[i-w] + z)
		p = x

		x = p + a[i]
		z = x - p
		s += (p - (x - z)) + (a[i] - z)
		p = x

		res[i-w+1] = (p + s) / float64(w)
	}

	return res
}

func sum2s_v2(a []float64, w int) []float64 {
	if len(a) < w {
		return nil
	}
	accum := a[0]
	resid := 0.0
	var m, p, q float64
	for i := 1; i < w; i++ {
		m = a[i]
		p = accum
		accum += m
		q = accum - p
		resid += (p - (accum - q)) + (m - q)
	}

	res := make([]float64, len(a)-w+1)
	res[0] = accum + resid
	var n, r, t float64
	for i := w; i < len(a); i++ {
		m = a[i-w]
		n = a[i]
		p = accum - m
		q = p - accum
		r = resid + ((accum - (p - q)) - (m + q))
		accum = p + n
		t = accum - p
		resid = r + ((p - (accum - t)) + (n - t))
		res[i-w+1] = accum + resid
	}

	return res
}
