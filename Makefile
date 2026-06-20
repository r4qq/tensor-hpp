default:
	g++ -O3 -march=x86-64-v3 -std=c++20 -fopt-info-vec -fopt-info-vec-missed test.cpp -o test

run:
	./test

perf:
	valgrind --tool=cachegrind ./test
