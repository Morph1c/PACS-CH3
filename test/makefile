CXX       = mpic++
CXXFLAGS ?= -std=c++17
# note if you create an include directory is to put all hpp files! You have left
# lapaceSover.hpp in src.
CPPFLAGS ?= -fopenmp -O3 -Wall -pedantic -Iinclude -I../include -I../src -I/usr/local/include/eigen3 

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
