# Laplace Matrix-Free Parallel Solver

This project implements a matrix-free parallel solver for the Laplace equation using the finite difference method. The solver is implemented in C++ and uses the Eigen library for matrix operations and MPI for parallelization and openMP directive for hybrid parallelization.

## Features

- Matrix-free implementation: The solver does not explicitly store the system matrix, which can save a significant amount of memory for large problems.
- Parallelization: The solver uses OpenMP to parallelize the iterations, which can significantly speed up the solution process on multi-core machines (remove comment in the local_solver, inside LaplaceSolver.cpp to abiltate openmp directive, test performance was run without due to multi-thread issues).
- Flexible boundary conditions: The solver can handle any function as the Dirichlet boundary condition.

## Usage
Clone the repository:
```shell
git clone git@github.com:Morph1c/PACS-CH2.git 
```
Then go to the src directory
```shell
cd test
```
run the test by simply:
```shell
chmod +x run_test.sh
./run_test.sh <n> 
```
where n are the number of rank which you want to test, this it will generate you a performance plot in /plots. While if you want to directly test by your self with mpirun then run the make:
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

## Performance result
![execution_times](https://github.com/Morph1c/PACS-CH3/assets/56799376/cd3567e0-6dba-436c-8d74-9534dabac8be)

