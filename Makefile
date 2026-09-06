default:
	g++ -O3 -fopenmp -march=native -std=c++23 -fopt-info-vec -fopt-info-vec-missed test.cpp -o test

run:
	./test