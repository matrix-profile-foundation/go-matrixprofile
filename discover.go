package matrixprofile

import (
	"errors"
	"math"
	"sort"

	"github.com/matrix-profile-foundation/go-matrixprofile/util"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/fourier"
)

// MotifGroup stores a list of indices representing a similar motif along
// with the minimum distance that this set of motif composes of.
type MotifGroup struct {
	Idx     []int
	MinDist float64
}

// DiscoverMotifs will iteratively go through the matrix profile to find the
// top k motifs with a given radius. Only applies to self joins.
func (mp MatrixProfile) DiscoverMotifs(k int, r float64) ([]MotifGroup, error) {
	if !mp.SelfJoin {
		return nil, errors.New("can only find top motifs if a self join is performed")
	}
	var err error
	var minDistIdx int

	motifs := make([]MotifGroup, k)

	mpCurrent, err := mp.applyAV()
	if err != nil {
		return nil, err
	}

	prof := make([]float64, len(mpCurrent)) // stores minimum matrix profile distance between motif pairs
	fft := fourier.NewFFT(mp.N)
	var j int

	for j = 0; j < k; j++ {
		// find minimum distance and index location
		motifDistance := math.Inf(1)
		minIdx := math.MaxInt64
		for i, d := range mpCurrent {
			if d < motifDistance {
				motifDistance = d
				minIdx = i
			}
		}

		if minIdx == math.MaxInt64 {
			// can't find any more motifs so returning what we currently found
			return motifs, nil
		}

		// filter out all indexes that have a distance within r*motifDistance
		motifSet := make(map[int]struct{})
		initialMotif := []int{minIdx, mp.Idx[minIdx]}
		motifSet[minIdx] = struct{}{}
		motifSet[mp.Idx[minIdx]] = struct{}{}

		if err = mp.distanceProfile(initialMotif[0], prof, fft); err != nil {
			return nil, err
		}

		// kill off any indices around the initial motif pair since they are
		// trivial solutions
		util.ApplyExclusionZone(prof, initialMotif[0], mp.M/2)
		util.ApplyExclusionZone(prof, initialMotif[1], mp.M/2)
		if j > 0 {
			for k := j; k >= 0; k-- {
				for _, idx := range motifs[k].Idx {
					util.ApplyExclusionZone(prof, idx, mp.M/2)
				}
			}
		}
		// keep looking for the closest index to the current motif. Each
		// index found will have an exclusion zone applied as to remove
		// trivial solutions. This eventually exits when there's nothing
		// found within the radius distance.
		for {
			minDistIdx = floats.MinIdx(prof)

			if prof[minDistIdx] < motifDistance*r {
				motifSet[minDistIdx] = struct{}{}
				util.ApplyExclusionZone(prof, minDistIdx, mp.M/2)
			} else {
				// the closest distance in the profile is greater than the desired
				// distance so break
				break
			}
		}

		// store the found motif indexes and create an exclusion zone around
		// each index in the current matrix profile
		motifs[j] = MotifGroup{
			Idx:     make([]int, 0, len(motifSet)),
			MinDist: motifDistance,
		}
		for idx := range motifSet {
			motifs[j].Idx = append(motifs[j].Idx, idx)
			util.ApplyExclusionZone(mpCurrent, idx, mp.M/2)
		}

		// sorts the indices in ascending order
		sort.IntSlice(motifs[j].Idx).Sort()
	}

	return motifs[:j], nil
}

// DiscoverDiscords finds the top k time series discords starting indexes from a computed
// matrix profile. Each discovery of a discord will apply an exclusion zone around
// the found index so that new discords can be discovered.
func (mp MatrixProfile) DiscoverDiscords(k int, exclusionZone int) ([]int, error) {
	mpCurrent, err := mp.applyAV()
	if err != nil {
		return nil, err
	}

	// if requested k is larger than length of the matrix profile, cap it
	if k > len(mpCurrent) {
		k = len(mpCurrent)
	}

	discords := make([]int, k)
	var maxVal float64
	var maxIdx int
	var i int

	for i = 0; i < k; i++ {
		maxVal = 0
		maxIdx = math.MaxInt64
		for j, val := range mpCurrent {
			if !math.IsInf(val, 1) && val > maxVal {
				maxVal = val
				maxIdx = j
			}
		}

		if maxIdx == math.MaxInt64 {
			break
		}

		discords[i] = maxIdx
		util.ApplyExclusionZone(mpCurrent, maxIdx, exclusionZone)
	}
	return discords[:i], nil
}

// DiscoverSegments finds the the index where there may be a potential timeseries
// change. Returns the index of the potential change, value of the corrected
// arc curve score and the histogram of all the crossings for each index in
// the matrix profile index. This approach is based on the UCR paper on
// segmentation of timeseries using matrix profiles which can be found
// https://www.cs.ucr.edu/%7Eeamonn/Segmentation_ICDM.pdf
func (mp MatrixProfile) DiscoverSegments() (int, float64, []float64) {
	histo := util.ArcCurve(mp.Idx)

	for i := 0; i < len(histo); i++ {
		if i == 0 || i == len(histo)-1 {
			histo[i] = math.Min(1.0, float64(len(histo)))
		} else {
			histo[i] = math.Min(1.0, histo[i]/util.Iac(float64(i), len(histo)))
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
