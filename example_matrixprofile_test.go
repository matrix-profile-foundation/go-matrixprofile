package matrixprofile

import (
	"fmt"

	"github.com/matrix-profile-foundation/go-matrixprofile/siggen"
)

func ExampleMatrixProfile_DiscoverSegments() {
	// generate a signal mainly composed of sine waves and switches
	// frequencies, amplitude, and offset midway through

	// amplitude of 1, frequency of 5Hz, sampling frequency of 100 Hz,
	// time of 2 seconds
	sin := siggen.Sin(1, 5, 0, 0, 100, 2)

	// amplitude of 0.25, frequency of 10Hz, offset of 0.75, sampling
	// frequency of 100 Hz, time of 1 second
	sin2 := siggen.Sin(0.25, 10, 0, 0.75, 100, 1)
	sig := siggen.Append(sin, sin2)

	// noise with an amplitude of 0.1
	noise := siggen.Noise(0.01, len(sig))
	sig = siggen.Add(sig, noise)

	// create a new MatrixProfile struct using the signal and a
	// subsequence length of 32. The second subsequence is set to nil
	// so we perform a self join.
	mp, err := New(sig, nil, 32)
	if err != nil {
		panic(err)
	}

	// run the STMP algorithm with self join. The matrix profile
	// will be stored in mp.MP and the matrix profile index will
	// be stored in mp.Idx
	o := NewMPOpts()
	o.Algorithm = AlgoSTMP

	if err = mp.Compute(o); err != nil {
		panic(err)
	}

	// segment the timeseries using the number of arc crossings over
	// each index in the matrix profile index
	idx, cac, _ := mp.DiscoverSegments()
	fmt.Printf("Signal change foud at index: %d\n", idx)
	fmt.Printf("Corrected Arc Curve (CAC) value: %.3f\n", cac)

	// Output:
	// Signal change foud at index: 194
	// Corrected Arc Curve (CAC) value: 0.000
}

func ExampleMatrixProfile_DiscoverMotifs() {
	// generate a signal mainly composed of sine waves and switches
	// frequencies, amplitude, and offset midway through

	// amplitude of 1, frequency of 5Hz, sampling frequency of 100 Hz,
	// time of 2 seconds
	sin := siggen.Sin(1, 5, 0, 0, 100, 2)

	// amplitude of 0.25, frequency of 10Hz, offset of 0.75, sampling
	// frequency of 100 Hz, time of 1 second
	sin2 := siggen.Sin(0.25, 10, 0, 0.75, 100, 1)
	sig := siggen.Append(sin, sin2)

	// create a new MatrixProfile struct using the signal and a
	// subsequence length of 32. The second subsequence is set to nil
	// so we perform a self join.
	mp, err := New(sig, nil, 32)
	if err != nil {
		panic(err)
	}

	// run the STMP algorithm with self join. The matrix profile
	// will be stored in mp.MP and the matrix profile index will
	// be stored in mp.Idx
	o := NewMPOpts()
	o.Algorithm = AlgoSTMP

	if err = mp.Compute(o); err != nil {
		panic(err)
	}

	// finds the top 3 motifs in the signal. Motif groups include
	// all subsequences that are within 2 times the distance of the
	// original motif pair
	motifs, err := mp.DiscoverMotifs(2, 2)
	if err != nil {
		panic(err)
	}

	for i, mg := range motifs {
		fmt.Printf("Motif Group %d\n", i)
		fmt.Printf("  %d motifs\n", len(mg.Idx))
	}

	// Output:
	// Motif Group 0
	//   2 motifs
	// Motif Group 1
	//   2 motifs
}
