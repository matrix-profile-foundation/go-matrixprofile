package matrixprofile

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
)

// PMP represents the pan matrix profile
type PMP struct {
	A        []float64   // query time series
	B        []float64   // timeseries to perform full join with
	SelfJoin bool        // indicates whether a self join is performed with an exclusion zone
	PMP      [][]float64 // pan matrix profile
	PIdx     [][]int     // pan matrix profile index
	PWindows []int       // pan matrix windows used and is aligned with PMP and PIdx
}

// NewPMP creates a new Pan matrix profile
func NewPMP(a, b []float64) (*PMP, error) {
	if a == nil || len(a) == 0 {
		return nil, fmt.Errorf("first slice is nil or has a length of 0")
	}

	if b != nil && len(b) == 0 {
		return nil, fmt.Errorf("second slice must be nil for self-join operation or have a length greater than 0")
	}

	p := PMP{A: a}
	if b == nil {
		p.B = a
		p.SelfJoin = true
	} else {
		p.B = b
	}

	return &p, nil
}

// Save will save the current matrix profile struct to disk
func (p PMP) Save(filepath, format string) error {
	var err error
	switch format {
	case "json":
		f, err := os.Open(filepath)
		if err != nil {
			f, err = os.Create(filepath)
			if err != nil {
				return err
			}
		}
		defer f.Close()
		out, err := json.Marshal(p)
		if err != nil {
			return err
		}
		_, err = f.Write(out)
	default:
		return fmt.Errorf("invalid save format, %s", format)
	}
	return err
}

// Load will attempt to load a matrix profile from a file for iterative use
func (p *PMP) Load(filepath, format string) error {
	var err error
	switch format {
	case "json":
		f, err := os.Open(filepath)
		if err != nil {
			return err
		}
		defer f.Close()
		b, err := ioutil.ReadAll(f)
		if err != nil {
			return err
		}
		err = json.Unmarshal(b, p)
	default:
		return fmt.Errorf("invalid load format, %s", format)
	}
	return err
}
