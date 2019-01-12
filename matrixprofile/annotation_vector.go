package matrixprofile

import (
	"math"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/stat"
)

// MakeCompexityAV creates an annotation vector that is based on the complexity
// estimation of the signal.
func MakeCompexityAV(d []float64, m int) []float64 {
	av := make([]float64, len(d)-m+1)
	var ce, minAV, maxAV float64
	minAV = math.Inf(1)
	maxAV = math.Inf(-1)
	for i := 0; i < len(d)-m+1; i++ {
		ce = 0.0
		for j := 1; j < m; j++ {
			ce += (d[i+j] - d[i+j-1]) * (d[i+j] - d[i+j-1])
		}
		av[i] = math.Sqrt(ce)
		if av[i] < minAV {
			minAV = av[i]
		}
		if av[i] > maxAV {
			maxAV = av[i]
		}
	}
	for i := 0; i < len(d)-m+1; i++ {
		if maxAV == 0 {
			av[i] = 0
		} else {
			av[i] = (av[i] - minAV) / maxAV
		}
	}

	return av
}

// MakeMeanStdAV creates an annotation vector by setting any values above the mean
// of the standard deviation vector to 0 and below to 1.
func MakeMeanStdAV(d []float64, m int) []float64 {
	av := make([]float64, len(d)-m+1)
	_, std, _ := movmeanstd(d, m)
	mu := stat.Mean(std, nil)
	for i := 0; i < len(d)-m+1; i++ {
		if std[i] < mu {
			av[i] = 1
		}
	}
	return av
}

// MakeClippingAV creates an annotation vector by setting subsequences with more
// clipping on the positive or negative side of the signal to lower importance.
func MakeClippingAV(d []float64, m int) []float64 {
	av := make([]float64, len(d)-m+1)
	maxVal, minVal := floats.Max(d), floats.Min(d)
	var numClip int
	for i := 0; i < len(d)-m+1; i++ {
		numClip = 0
		for j := 0; j < m; j++ {
			if d[i+j] == maxVal || d[i+j] == minVal {
				numClip++
			}
		}
		av[i] = 1.0 - (float64(numClip)-minVal)/(maxVal-minVal)
	}
	return av
}
