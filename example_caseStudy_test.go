package matrixprofile

import (
	"fmt"

	"github.com/matrix-profile-foundation/go-matrixprofile/siggen"
)

func Example_caseStudy() {
	sin := siggen.Sin(1, 5, 0, 0, 100, 2)
	sin2 := siggen.Sin(0.25, 10, 0, 0.75, 100, 0.25)
	saw := siggen.Sawtooth(0.5, 7, 0, 0, 100, 1)
	line := siggen.Line(0, 0, len(sin2)*4)
	sig := siggen.Append(sin, sin2, sin, line, sin2, line, sin2, line, saw)

	noise := siggen.Noise(0.1, len(sig))
	sig = siggen.Add(sig, noise)

	var m int
	m = 32
	mp, err := New(sig, nil, m)
	if err != nil {
		panic(err)
	}

	ao := NewAnalyzeOpts()
	ao.OutputFilename = "mp_sine.png"

	if err = mp.Analyze(nil, ao); err != nil {
		panic(err)
	}

	fmt.Printf("Saved png file result to %s\n", ao.OutputFilename)
	// Output: Saved png file result to mp_sine.png
}

func Example_kDimensionalCaseStudy() {
	sin := siggen.Sin(1, 4, 0, 0, 100, 0.25)
	saw := siggen.Sawtooth(1, 4, 0, 0, 100, 0.25)
	square := siggen.Square(1, 4, 0, 0, 100, 0.25)
	line := siggen.Line(0, 0, len(sin)*4)
	line2 := siggen.Line(0, 0, len(sin)*3)
	sig := make([][]float64, 3)
	sig[0] = siggen.Append(line, line, line, saw, line2, saw, line2)
	sig[1] = siggen.Append(line, sin, line2, sin, line2, sin, line2, sin, line2)
	sig[2] = siggen.Append(line, square, line2, square, line2, square, line2, square, line2)

	sig[0] = siggen.Add(sig[0], siggen.Noise(0.1, len(sig[0])))
	sig[1] = siggen.Add(sig[1], siggen.Noise(0.1, len(sig[0])))
	sig[2] = siggen.Add(sig[2], siggen.Noise(0.1, len(sig[0])))

	m := 25
	mp, err := NewKMP(sig, m)
	if err != nil {
		panic(err)
	}

	if err = mp.Compute(); err != nil {
		panic(err)
	}

	if err = mp.Visualize("mp_kdim.png"); err != nil {
		panic(err)
	}

	fmt.Println("Saved png file result to mp_kdim.png")
	// Output: Saved png file result to mp_kdim.png
}
