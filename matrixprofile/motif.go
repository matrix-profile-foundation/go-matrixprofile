package matrixprofile

import (
	"errors"
	"math"
	"sort"
)

// MotifGroup stores a list of indices representing a similar motif along with the minimum distance that this set of motif composes of
type MotifGroup struct {
	Idx     []int
	MinDist float64
}

// TopKMotifs will iteratively go through the matrix profile to find the top k motifs with a given radius. Only applies to self joins
func (mp MatrixProfile) TopKMotifs(k int, r float64) ([]MotifGroup, error) {
	if !mp.selfJoin {
		return nil, errors.New("can only find top motifs if a self join is performed")
	}

	motifs := make([]MotifGroup, k)

	mpCurrent := make([]float64, len(mp.MP))
	copy(mpCurrent, mp.MP)

	for j := 0; j < k; j++ {
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

		for _, idx := range initialMotif {
			prof, err := mp.distanceProfile(idx)
			if err != nil {
				return nil, err
			}
			for i, d := range prof {
				if d < motifDistance*r {
					motifSet[i] = struct{}{}
				}
			}
		}

		// store the found motif indexes and create an exclusion zone around each index in the current matrix profile
		motifs[j] = MotifGroup{
			Idx:     make([]int, 0, len(motifSet)),
			MinDist: motifDistance,
		}
		for idx, _ := range motifSet {
			motifs[j].Idx = append(motifs[j].Idx, idx)
			applyExclusionZone(mpCurrent, idx, mp.m/2)
		}

		// sorts the indices in ascending order
		sort.IntSlice(motifs[j].Idx).Sort()
	}

	return motifs, nil
}
