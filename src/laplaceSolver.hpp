#ifndef LAPLACESOLVER_HPP
#define LAPLACESOLVER_HPP
#include <mpi.h>
#include <vector>
#include <cmath>
#include <omp.h>
#include <iostream>
#include <functional>
#include <chrono>
#include <iostream>
#include <Eigen/Dense>


namespace parallel_solver{

class LaplaceSolver{
private:
    int n;
    int it_max = 100000;
    double tol;
    double h;
    std::chrono::duration<double> diff;
    int rows_per_proc;
    std::function<double(double, double)> f;
    std::function<double(double, double)> g;
    std::function<double(double, double)> u_ex_fun;
    bool multiprocess;

    std::vector<int> rows_per_rank;
    std::vector<int> real_start_pos;

    // used for postprocessing only from rank 0
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> U;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Error_field;

    // local matrices used by each rank
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> local_U;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> local_F;

    

public:
    LaplaceSolver(int n, double tol, std::function<double(double, double)> f, std::function<double(double, double)> g_data,
 std::function<double(double, double)> _u_ex_fun, bool multiprocess) 
    : n(n), tol(tol), f(f), g(g_data), u_ex_fun(_u_ex_fun), multiprocess(multiprocess){
        U = Eigen::MatrixXd::Zero(n, n);
        Error_field = Eigen::MatrixXd::Zero(n, n);
        h = 1.0 / (n - 1);
    }


    double parallel_iter(int rank, int size);
    double local_solver_iter(int rank);
    void solve();
    void postprocess(const std::string& filename, const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& U, int nx, int ny, double dx, double dy);
    void assemble_local_matrix(int rank, int size);
    void assemble_local_F(int rank, int size);
    void assemble_matrix(int rank, int size);
    void boundary_conditions(int rank, int size);
    double get_execution_time();
};

}


#endif
