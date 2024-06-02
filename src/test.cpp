#include <iostream>
#include "../src/laplaceSolver.hpp"

int main(int argc, char** argv) {
    int n = atoi(argv[1]); // Number of grid points
    double tol = 1e-5;
    auto f = [](double x, double y) { return 8 * M_PI * M_PI * sin(2 * M_PI * x) * sin(2 * M_PI * y); };
    LaplaceSolver solver(n, tol, f);
    solver.solve(argc, argv);
    return 0;
}