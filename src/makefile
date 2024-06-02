CXX       = mpic++
CXXFLAGS ?= -std=c++17
CPPFLAGS ?= -fopenmp -O3 -Wall -pedantic

LDFLAGS ?=
LIBS    ?=

EXEC = main

.PHONY = all $(EXEC) clean distclean $(DEPEND)

all: $(EXEC)

$(EXEC): %: %.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) $< $(LIBS) -o $@

clean:
	$(RM) *.o
	$(RM) $(EXEC)
	$(RM) *.csv *.out *.bak *~

distclean: clean
	$(RM) $(EXEC)
	$(RM) *.csv *.out *.bak *~