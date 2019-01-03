usage:
	@echo "make all     : Runs all tests, examples, and benchmarks"
	@echo "make test    : Runs test suite"
	@echo "make bench   : Runs benchmarks"
	@echo "make example : Runs examples" 
	@echo "make setup   : Installs all needed dependencies"

all: test bench

test:
	go test -race -cover ./...

bench:
	go test ./... -run=XX -bench=. -test.benchmem

example:
	go test ./... -run=Example

setup:
	go get -u gonum.org/v1/gonum/...
	go get -u gonum.org/v1/plot/...
