package matrixprofile

import (
	"math"
)

// Segment finds the the index where there may be a potential timeseries change. This is based
// from the UCR paper on segmentation of timeseries using matrix profiles which can be found
// https://www.cs.ucr.edu/%7Eeamonn/Segmentation_ICDM.pdf
func (mp MatrixProfile) Segment() (int, float64, []float64) {
	histo := arcCurve(mp.Idx)

	for i := 0; i < len(histo); i++ {
		if i == 0 || i == len(histo)-1 {
			histo[i] = math.Min(1.0, float64(len(histo)))
		} else {
			histo[i] = math.Min(1.0, histo[i]/iac(float64(i), len(histo)))
		}
	}

	minIdx := math.MaxInt64
	minVal := math.Inf(1)
	for i := 0; i < len(histo); i++ {
		if histo[i] < minVal {
			minIdx = i
			minVal = histo[i]
		}
	}

	return minIdx, float64(minVal), histo
}

// arcCurve computes the arc curve (histogram) which is uncorrected for. This loops through
// the matrix profile index and increments the counter for each index that the destination
// index passes through start from the index in the matrix profile index.
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

// iac represents the ideal arc curve with a maximum of n/2 and 0 values at 0 and n-1.
// -(sqrt(2/n)*(x-n/2))^2 + n/2 = y is the derived equation to ensure the requirements.
func iac(x float64, n int) float64 {
	return -math.Pow(math.Sqrt(2/float64(n))*(x-float64(n)/2.0), 2.0) + float64(n)/2.0
}
