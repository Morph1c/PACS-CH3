# Laplace Matrix-Free Parallel Solver

This project implements a matrix-free parallel solver for the Laplace equation using the finite difference method. The solver is implemented in C++ and uses the Eigen library for matrix operations and OpenMP for parallelization.

## Features

- Matrix-free implementation: The solver does not explicitly store the system matrix, which can save a significant amount of memory for large problems.
- Parallelization: The solver uses OpenMP to parallelize the iterations, which can significantly speed up the solution process on multi-core machines.
- Flexible boundary conditions: The solver can handle any function as the boundary condition.

## Usage
Clone the repository:
```shell
git clone git@github.com:Morph1c/PACS-CH2.git 
```
Then go to the src directory
```shell
cd src
```
run the make:
```shell
make 
```
Then run the code using p processors and n grid points:
```shell
mpirun -np <p> ./test n
```
If you want to plot the execution time run:
```shell
python3 plot_times.py
```
Then look at plots folder for:

- Execution times plot
- Approximated solution in .vtk file
- Error filed in .vtk

