[![Build Status](https://travis-ci.com/aouyang1/go-matrixprofile.svg?branch=master)](https://travis-ci.com/aouyang1/go-matrixprofile)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# go-matrixprofile

Golang library for computing a matrix profiles and matrix profile indexes. Features also include time series segmentation and motif discovery after computing the matrix profile. https://godoc.org/github.com/aouyang1/go-matrixprofile

## Contents
- [Installation](#installation)
- [Quick start](#quick-start)
- [Benchmarks](#benchmarks)

## Installation
```sh
$ go get -u github.com/aouyang1/go-matrixprofile/matrixprofile
```

## Quick start
```sh
$ cat example_mp.go
```
```go
package main

import (
	"fmt"

	"github.com/aouyang1/go-matrixprofile/matrixprofile"
)

func main() {
	sig := []float64{0, 0.99, 1, 0, 0, 0.98, 1, 0, 0, 0.96, 1, 0}

	mp, err := matrixprofile.New(sig, nil, 4)
	if err != nil {
		panic(err)
	}

	if err = mp.Stmp(); err != nil {
		panic(err)
	}

	fmt.Printf("Signal:         %.3f\n", sig)
	fmt.Printf("Matrix Profile: %.3f\n", mp.MP)
	fmt.Printf("Profile Index:  %5d\n", mp.Idx)
}
```
```sh
$ go run example_mp.go
Signal:         [0.000 0.990 1.000 0.000 0.000 0.980 1.000 0.000 0.000 0.960 1.000 0.000]
Matrix Profile: [0.014 0.014 0.029 0.029 0.014 0.014 0.029 0.029 0.029]
Profile Index:  [    4     5     6     7     0     1     2     3     4]
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

