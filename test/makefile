CXX       = mpic++
CXXFLAGS ?= -std=c++17
CPPFLAGS ?= -fopenmp -O3 -Wall -pedantic -Iinclude -I "/u/sw/toolchains/gcc-glibc/11.2.0/pkgs/eigen/3.3.9/include/eigen3" 

LDFLAGS ?=
LIBS    ?=

EXEC = test
SRCS = test.cpp ../src/laplaceSolver.cpp
OBJS = $(SRCS:.cpp=.o)

.PHONY = all $(EXEC) clean $(DEPEND)

all: $(EXEC)

$(EXEC): $(OBJS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) $^ $(LIBS) -o $@

%.o: %.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

clean:
	$(RM) *.o ../src/*.o
	$(RM) $(EXEC)
	$(RM) *.csv *.out *.bak *.vtk *~