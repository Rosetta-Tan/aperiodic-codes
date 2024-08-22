CXX ?= clang++
CXXFLAGS ?= -std=c++20 -O3 -Wall -Wno-unused-variable -Wno-unused-but-set-variable
LDFLAGS ?= -lpthread -lm -lz -ldl -lhighs

cut_multi: cut_multi.cc cnpy.cpp
	$(CXX) -m64 $(CXXFLAGS) $^ -o $@ -I/usr/local/include/highs -L/usr/local/lib -Wl,-rpath /usr/local/lib $(LDFLAGS)

clean:
	rm -f cut_multi
