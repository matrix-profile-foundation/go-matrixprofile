// Package av generates a set of annotation vectors that can be applied onto a matrix profile before computing other features such as motifs, discords, etc.
package av

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/stat"

	"github.com/matrix-profile-foundation/go-matrixprofile/util"
)

type AV string

const (
	Default    AV = "default"    // Default is the default annotation vector of all ones
	Complexity AV = "complexity" // Complexity is the annotation vector that focuses on areas of high "complexity"
	MeanStd    AV = "mean_std"   // MeanStd is the annotation vector focusing on areas where the signal is within a standard deviation of the mean
	Clipping   AV = "clipping"   // Clipping is the annotation vector reducing the importance of areas showing clipping effects on the positive and negative regime
)

// Create returns the annotation vector given an input time series and a window size m
func Create(av AV, ts []float64, m int) ([]float64, error) {
	var avec []float64
	switch av {
	case Default:
		avec = makeDefault(ts, m)
	case Complexity:
		avec = makeCompexity(ts, m)
	case MeanStd:
		avec = makeMeanStd(ts, m)
	case Clipping:
		avec = makeClipping(ts, m)
	default:
		return nil, fmt.Errorf("invalid annotation vector specified with matrix profile, %s", av)
	}
	return avec, nil
}

// makeDefault creates a default annotation vector of all ones resulting in
// no change to the matrix profile when applied
func makeDefault(d []float64, m int) []float64 {
	av := make([]float64, len(d)-m+1)
	for i := 0; i < len(av); i++ {
		av[i] = 1.0
	}
	return av
}

// makeCompexity creates an annotation vector that is based on the complexity
// estimation of the signal.
func makeCompexity(d []float64, m int) []float64 {
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

// makeMeanStd creates an annotation vector by setting any values above the mean
// of the standard deviation vector to 0 and below to 1.
func makeMeanStd(d []float64, m int) []float64 {
	av := make([]float64, len(d)-m+1)
	_, std, _ := util.MovMeanStd(d, m)
	mu := stat.Mean(std, nil)
	for i := 0; i < len(d)-m+1; i++ {
		if std[i] < mu {
			av[i] = 1
		}
	}
	return av
}

// makeClipping creates an annotation vector by setting subsequences with more
// clipping on the positive or negative side of the signal to lower importance.
func makeClipping(d []float64, m int) []float64 {
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
		av[i] = float64(numClip)
	}

	minVal = floats.Min(av)
	for i := 0; i < len(av); i++ {
		av[i] -= minVal
	}

	maxVal = floats.Max(av)
	for i := 0; i < len(av); i++ {
		av[i] = 1 - av[i]/maxVal
	}

	return av
}
