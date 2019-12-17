package matrixprofile

import (
	"fmt"
)

func Example() {
	sig := []float64{0, 0.99, 1, 0, 0, 0.98, 1, 0, 0, 0.96, 1, 0}

	mp, err := New(sig, nil, 4)
	if err != nil {
		panic(err)
	}

	if err = mp.Compute(NewOptions()); err != nil {
		panic(err)
	}

	fmt.Printf("Signal:         %.3f\n", sig)
	fmt.Printf("Matrix Profile: %.3f\n", mp.MP)
	fmt.Printf("Profile Index:  %5d\n", mp.Idx)

	// Output:
	// Signal:         [0.000 0.990 1.000 0.000 0.000 0.980 1.000 0.000 0.000 0.960 1.000 0.000]
	// Matrix Profile: [0.014 0.014 0.029 0.029 0.014 0.014 0.029 0.029 0.029]
	// Profile Index:  [    4     5     6     7     0     1     2     3     4]
}
