package matrixprofile

import (
	"os"
	"testing"
)

func TestPMPSave(t *testing.T) {
	ts := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}
	p, err := NewPMP(ts, nil)
	p.Compute(NewPMPComputeOpts(3, 5))
	filepath := "./mp.json"
	err = p.Save(filepath, "json")
	if err != nil {
		t.Errorf("Received error while saving matrix profile, %v", err)
	}
	if err = os.Remove(filepath); err != nil {
		t.Errorf("Could not remove file, %s, %v", filepath, err)
	}
}

func TestPMPLoad(t *testing.T) {
	ts := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}
	p, err := NewPMP(ts, nil)
	p.Compute(NewPMPComputeOpts(3, 5))
	filepath := "./mp.json"
	if err = p.Save(filepath, "json"); err != nil {
		t.Errorf("Received error while saving matrix profile, %v", err)
	}

	newP := &PMP{}
	if err = newP.Load(filepath, "json"); err != nil {
		t.Errorf("Failed to load %s, %v", filepath, err)
	}

	if err = os.Remove(filepath); err != nil {
		t.Errorf("Could not remove file, %s, %v", filepath, err)
	}

	if len(newP.A) != len(ts) {
		t.Errorf("Expected timeseries length of %d, but got %d", len(ts), len(newP.A))
	}

}
