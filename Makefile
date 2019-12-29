usage:
	@echo "make all       : Runs all tests, examples, and benchmarks"
	@echo "make test      : Runs test suite"
	@echo "make bench     : Runs benchmarks"
	@echo "make example   : Runs examples"
	@echo "make travis-ci : Travis CI specific testing"

all: test bench example

test:
	go test -race -cover -run=Test ./...

bench:
	go test ./... -run=NONE -bench=. -test.benchmem > new_bench.txt
	go get golang.org/x/tools/cmd/benchcmp
	benchcmp curr_bench.txt new_bench.txt

example:
	go test ./... -run=Example

travis-ci:
	go test -v ./... -race -coverprofile=coverage.txt -covermode=atomic
