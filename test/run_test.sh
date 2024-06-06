#!/bin/bash

# Clean the build
make clean

# Build the project
make

# Run the MPI program
mpirun -np $1 ./test

# Run the Python script
python3 plot_times.py
