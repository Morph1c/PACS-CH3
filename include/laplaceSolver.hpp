#ifndef LAPLACESOLVER_HPP
#define LAPLACESOLVER_HPP
#include <mpi.h>
#include <vector>
#include <cmath>
#include <omp.h>
#include <iostream>
#include <functional>
#include <Eigen/Dense>


namespace parallel_solver{

class LaplaceSolver{
private:
    int n;
    int it_max = 1000;
    double tol;
    double h;
    int rows_per_proc;
    std::function<double(double, double)> f;

    //std::vector<std::vector<double>> U;
    //std::vector<std::vector<double>> U_exact;

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> U;

public:
    LaplaceSolver(int n, double tol, std::function<double(double, double)> f) : n(n), tol(tol), f(f) {
        //U = std::vector<std::vector<double>>(n, std::vector<double>(n, 0.0));
        //U_exact = std::vector<std::vector<double>>(n, std::vector<double>(n, 0.0));
        U = Eigen::MatrixXd::Zero(n, n);
        h = 1.0 / (n - 1);
    }

    double parallel_iter(int rank, int size);
    double local_solver_iter(int start_pos, int end_pos);
    void solve(int argc, char** argv);
    void postprocess(const std::string& filename, const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& U, int nx, int ny, double dx, double dy);
    void assemble_matrix(int rank);
};

}


#endif