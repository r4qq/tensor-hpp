default:
	g++ -O3 -march=native -std=c++20 -fopt-info-vec -fopt-info-vec-missed test.cpp -o test

run:
	./test