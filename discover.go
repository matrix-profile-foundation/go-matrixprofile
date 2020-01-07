package matrixprofile

import (
	"math"
)

// MotifGroup stores a list of indices representing a similar motif along
// with the minimum distance that this set of motif composes of.
type MotifGroup struct {
	Idx     []int
	MinDist float64
}

// arcCurve computes the arc curve (histogram) which is uncorrected for.
// This loops through the matrix profile index and increments the
// counter for each index that the destination index passes through
// start from the index in the matrix profile index.
func arcCurve(mpIdx []int) []float64 {
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

// iac represents the ideal arc curve with a maximum of n/2 and 0 values
// at 0 and n-1. The derived equation to ensure the requirements is
// -(sqrt(2/n)*(x-n/2))^2 + n/2 = y
func iac(x float64, n int) float64 {
	return -math.Pow(math.Sqrt(2/float64(n))*(x-float64(n)/2.0), 2.0) + float64(n)/2.0
}
