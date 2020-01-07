package matrixprofile

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"math"
	"os"

	"github.com/matrix-profile-foundation/go-matrixprofile/util"
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

// PMPComputeOptions are parameters to vary the algorithm to compute the pan matrix profile.
type PMPComputeOptions struct {
	LowerM int // used for pan matrix profile
	UpperM int // used for pan matrix profile
	Opts   *ComputeOptions
}

// NewPMPComputeOpts returns a default PMPComputeOptions
func NewPMPComputeOpts(l, u int) *PMPComputeOptions {
	if l > u {
		u = l
	}
	return &PMPComputeOptions{
		LowerM: l,
		UpperM: u,
		Opts:   NewComputeOpts(),
	}
}

// Compute calculate the pan matrixprofile given a set of input options.
func (p *PMP) Compute(o *PMPComputeOptions) error {
	if o == nil {
		return errors.New("Must provide PMP compute options")
	}

	return p.pmp(o)
}

func (p *PMP) pmp(o *PMPComputeOptions) error {
	windows := util.BinarySplit(o.LowerM, o.UpperM)
	windows = windows[:int(float64(len(windows))*o.Opts.Sample)]
	if len(windows) < 1 {
		return errors.New("Need more than one subsequence window for pmp")
	}
	p.PWindows = windows

	p.PMP = make([][]float64, len(windows))
	p.PIdx = make([][]int, len(windows))
	for i, m := range windows {
		lenA := len(p.A) - m + 1
		p.PMP[i] = make([]float64, lenA)
		p.PIdx[i] = make([]int, lenA)
		for j := 0; j < lenA; j++ {
			p.PMP[i][j] = math.Inf(1)
			p.PIdx[i][j] = math.MaxInt64
		}
	}

	// need to create a new mp
	var mp *MatrixProfile
	var err error
	if p.SelfJoin {
		mp, err = New(p.A, nil, windows[0])
	} else {
		mp, err = New(p.A, p.B, windows[0])
	}
	if err != nil {
		return err
	}

	for i, m := range windows {
		mp.M = m
		if err := mp.mpx(o.Opts); err != nil {
			return err
		}
		copy(p.PMP[i], mp.MP)
		copy(p.PIdx[i], mp.Idx)
	}

	return nil
}
