[![Build Status](https://travis-ci.com/aouyang1/go-matrixprofile.svg?branch=master)](https://travis-ci.com/aouyang1/go-matrixprofile)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# go-matrixprofile

Golang library for computing a matrix profiles and matrix profile indexes. Features also include time series segmentation and motif discovery after computing the matrix profile.

## Contents
- [Installation](#installation)
- [Quick start](#quick-start)
- [Benchmarks](#benchmarks)

## Installation
```sh
$ go get -u github.com/aouyang1/go-matrixprofile
```

## Quick start
```go
package main

import "github.com/aouyang1/go-matrixprofile"

func main() {
  // generate a synthetic signal to run a self join on
  sig := generateSignal()

	var m, k int
	var r float64
	m = 32 // subsequence length
	k = 6  // find the top N motifs
	r = 3  // motif groups contain subsequences that are at most R time distance 
         // from the initial motif

  // creates a matrix profile struct
	mp, err := matrixprofile.New(sig, nil, m)
	if err != nil {
		panic(err)
	}

  // computes the matrix profile and matrix profile index using the STMP algorithm
	if err = mp.Stmp(); err != nil {
		panic(err)
	}

  // uses the matrix profile index to compute the corrected arc curve
	_, _, cac := mp.Segment()

  // uses the matrix profile and matrix profile index to find the top K motif groups
  // within a radius of r times the minimum distance in the motif group
	motifs, err := mp.TopKMotifs(k, r)
	if err != nil {
		panic(err)
	}
}

func generateSignal() []float64 {
  // Amp: 1, Freq: 5Hz, Sampling Freq: 100Hz, Duration: 2sec
	sin := matrixprofile.Sin(1, 5, 0, 0, 100, 2)

  // Amp: 0.25, Freq: 10Hz, Offset: 0.75, Sampling Freq: 100Hz, Duration: 0.25sec
	sin2 := matrixprofile.Sin(0.25, 10, 0, 0.75, 100, 0.25)

  // Amp: 0.3, Duration: 1sec
	noise := matrixprofile.Noise(0.3, len(sin2)*4)

	sig := append(sin, sin2...)
	sig = append(sig, noise...)
	sig = append(sig, sin2...)
	sig = append(sig, noise...)
	sig = append(sig, sin2...)
	sig = append(sig, noise...)

  // Add additional noise to the entire signal
	noise = matrixprofile.Noise(0.1, len(sig))
	sig = matrixprofile.SigAdd(sig, noise)

  return sig
}
```

## Benchmarks
Benchmark name               | NumReps |    Time/Rep   |  Memory/Rep  |   Alloc/Rep
-----------------------------|--------:|--------------:|-------------:|--------------:
BenchmarkZNormalize-4        | 10000000|      178 ns/op|      256 B/op|    1 allocs/op
BenchmarkMovstd-4            |   200000|    11908 ns/op|    27136 B/op|    3 allocs/op
BenchmarkCrossCorrelate-4    |    20000|    66093 ns/op|    34053 B/op|    4 allocs/op
BenchmarkMass-4              |    20000|    68604 ns/op|    42501 B/op|    6 allocs/op
BenchmarkDistanceProfile-4   |    20000|    66735 ns/op|    42501 B/op|    6 allocs/op
BenchmarkStmp/m16-4          |       20| 71546986 ns/op| 42202424 B/op| 5958 allocs/op
BenchmarkStmp/m32-4          |       20| 67491284 ns/op| 42202424 B/op| 5958 allocs/op
BenchmarkStmp/m64-4          |       20| 72172709 ns/op| 42202424 B/op| 5958 allocs/op
BenchmarkStmp/m128-4         |       20| 68277992 ns/op| 42202424 B/op| 5958 allocs/op

