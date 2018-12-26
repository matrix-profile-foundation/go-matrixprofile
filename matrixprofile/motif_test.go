package matrixprofile

import (
	"math"
	"testing"
)

func TestTopKMotifs(t *testing.T) {
	a := []float64{0, 0, 0.56, 0.99, 0.97, 0.75, 0, 0, 0, 0.43, 0.98, 0.99, 0.65, 0, 0, 0, 0.6, 0.97, 0.965, 0.8, 0, 0, 0}
	a = sigAdd(a, generateNoise(1e-7, len(a)))

	expectedMotifs := [][]int{{1, 15}, {0, 7, 14}, {3, 10}}
	expectedMinDist := []float64{0.1459618197766371, 0.3352336136782056, 0.46369664551715467}
	mp, err := New(a, nil, 7)
	if err != nil {
		t.Error(err)
	}
	if err = mp.Stmp(); err != nil {
		t.Error(err)
	}
	motifs, err := mp.TopKMotifs(3, 2)

	for i, mg := range motifs {
		if len(mg.Idx) != len(expectedMotifs[i]) {
			t.Errorf("expected %d motifs for group %d, but got %d", len(expectedMotifs[i]), i, len(mg.Idx))
		}

		for j, idx := range mg.Idx {
			if idx != expectedMotifs[i][j] {
				t.Errorf("expected index, %d for group %d, but got %d", expectedMotifs[i][j], i, idx)
			}
		}
		if math.Abs(mg.MinDist-expectedMinDist[i]) > 1e-7 {
			t.Errorf("expected minimum distance, %.3f for group %d, but got %.3f", expectedMinDist[i], i, mg.MinDist)
		}
	}
}
