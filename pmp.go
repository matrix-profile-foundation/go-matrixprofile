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
	A        []float64   `json:"a"`         // query time series
	B        []float64   `json:"b"`         // timeseries to perform full join with
	SelfJoin bool        `json:"self_join"` // indicates whether a self join is performed with an exclusion zone
	PMP      [][]float64 `json:"pmp"`       // pan matrix profile
	PIdx     [][]int     `json:"ppi"`       // pan matrix profile index
	PWindows []int       `json:"windows"`   // pan matrix windows used and is aligned with PMP and PIdx
	Opts     *PMPOptions `json:"options"`   // options used for the computation
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

// PMPOptions are parameters to vary the algorithm to compute the pan matrix profile.
type PMPOptions struct {
	LowerM int        `json:"lower_m"` // used for pan matrix profile
	UpperM int        `json:"upper_m"` // used for pan matrix profile
	MPOpts *MPOptions `json:"mp_options"`
}

// NewPMPOpts returns a default PMPOptions
func NewPMPOpts(l, u int) *PMPOptions {
	if l > u {
		u = l
	}
	return &PMPOptions{
		LowerM: l,
		UpperM: u,
		MPOpts: NewMPOpts(),
	}
}

// Compute calculate the pan matrixprofile given a set of input options.
func (p *PMP) Compute(o *PMPOptions) error {
	if o == nil {
		return errors.New("Must provide PMP compute options")
	}
	p.Opts = o
	return p.pmp()
}

func (p *PMP) pmp() error {
	windows := util.BinarySplit(p.Opts.LowerM, p.Opts.UpperM)
	windows = windows[:int(float64(len(windows))*p.Opts.MPOpts.Sample)]
	if len(windows) < 1 {
		return errors.New("Need more than one subsequence window for pmp")
	}
	p.PWindows = windows

	p.PMP = make([][]float64, len(windows))
	p.PIdx = make([][]int, len(windows))
	for i := 0; i < len(windows); i++ {
		lenA := len(p.A) - (i + p.Opts.LowerM) + 1
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

	for _, w := range windows {
		mp.W = w
		if err := mp.Compute(p.Opts.MPOpts); err != nil {
			return err
		}
		copy(p.PMP[w-p.Opts.LowerM], mp.MP)
		copy(p.PIdx[w-p.Opts.LowerM], mp.Idx)
	}

	return nil
}

// Analyze has not been implemented yet
func (p PMP) Analyze(co *MPOptions, ao *AnalyzeOptions) error {
	return errors.New("Analyze for PMP has not been implemented yet.")
}

// DiscoverMotifs has not been implemented yet
func (p PMP) DiscoverMotifs(k int, r float64) ([]MotifGroup, error) {
	return nil, errors.New("Motifs for PMP has not been implemented yet.")
}

// DiscoverDiscords has not been implemented yet
func (p PMP) DiscoverDiscords(k int, exclusionZone int) ([]int, error) {
	return nil, errors.New("Discords for PMP has not been implemented yet.")
}

// DiscoverSegments has not been implemented yet
func (p PMP) DiscoverSegments() (int, float64, []float64) {
	return 0, 0, nil
}

// Visualize has not been implemented yet
func (p PMP) Visualize(fn string, motifs []MotifGroup, discords []int, cac []float64) error {
	return errors.New("Visualize for PMP has not been implemented yet.")
}
