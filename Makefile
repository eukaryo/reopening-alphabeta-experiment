enumerate: reversi6x6-reopening-ab.cpp
	g++ -I ./ -std=c++17 -fopenmp -static-libstdc++ -O2 -flto -march=native reversi6x6-reopening-ab.cpp -o reopening
clean:
	rm -f *.o reopening