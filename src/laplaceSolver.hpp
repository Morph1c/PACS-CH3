#include <mpi.h>
#include <vector>
#include <cmath>
#include <omp.h>
#include <iostream>
#include <functional>
#include "../include/writeVTK.hpp"

class LaplaceSolver {
private:
    int n;
    int it_max = 100000;
    double tol;
    double h;
    int rows_per_proc;
    std::function<double(double, double)> f;
    std::vector<std::vector<double>> U;
    std::vector<std::vector<double>> U_exact;

public:
    LaplaceSolver(int n, double tol, std::function<double(double, double)> f) : n(n), tol(tol), f(f) {
        U = std::vector<std::vector<double>>(n, std::vector<double>(n, 0.0));
        U_exact = std::vector<std::vector<double>>(n, std::vector<double>(n, 0.0));
        h = 1.0 / (n - 1);
    }

    double jacobi_iter(int rank, int size){
        // The code for one iteration of Jacobi goes here

        rows_per_proc = n / size;

        if (n % size != 0) {
            std::cerr << "The number of rows must be divisible by the number of processes" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        } 
    
        double norm;
        double res_k;

        int start_pos = (rank == 0) ? 1 : rank * rows_per_proc;
        int end_pos = (rank == size - 1) ? n - 1 : (rank + 1) * rows_per_proc;


        norm = 0.0;
        res_k = 0.0;

        //#pragma omp parallel for reduction(+:norm) collapse(2)
        for (int i = start_pos; i < end_pos; ++i) {
             // every rank modifies the first row, except the first rank
                                              // since in that case are boundary conditions
                                              // moreover each rank doesn't modify the last row
            for (int j = 1; j < n - 1; ++j) {
                double temp = U[i][j];
                U[i][j] = 0.25 * (U[i-1][j] + U[i+1][j] + U[i][j-1] + U[i][j+1] + h*h*f(i*h, j*h));
                norm += (U[i][j] - temp) * (U[i][j] - temp);
            }
        }

        // Send the last row to the next rank
        // Send the last row to the next rank and receive the first row from the next rank
        MPI_Request requests[4];
        int request_count = 0;

        if (rank < size - 1) {
            MPI_Isend(&U[(rank + 1) * rows_per_proc - 1][0], n, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &requests[request_count++]);
            MPI_Irecv(&U[(rank + 1) * rows_per_proc][0], n, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD, &requests[request_count++]);
        }
        

     // Receive the last row from the previous rank and send the first row to the previous rank
        if (rank > 0) {
            MPI_Isend(&U[rank * rows_per_proc][0], n, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, &requests[request_count++]);
            MPI_Irecv(&U[rank * rows_per_proc - 1][0], n, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &requests[request_count++]);
        }

        //MPI_Waitall(request_count, requests, MPI_STATUSES_IGNORE);

        MPI_Barrier(MPI_COMM_WORLD);  // Add this line

        //std::cout << "Rank: " << rank << " norm: " << norm << " iter " << it << std::endl;
        MPI_Allreduce(&norm, &res_k, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        // save the solution into U
    
        return res_k;

    }

    void solve(int argc, char** argv) {
        // The code for solving the problem goes here
         // Set boundary conditions
        MPI_Init(&argc, &argv);


        int size, rank;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        for (int i = 0; i < n; ++i) {
            U[0][i] = U[i][0] = U[n-1][i] = U[i][n-1] = 0.0;
            U_exact[0][i] = U_exact[i][0] = U_exact[n-1][i] = U_exact[i][n-1] = 0.0;
        }

        double error;
        for(size_t it = 0; it < it_max; ++it){
                error = jacobi_iter(rank, size);
                std::cout << "Rank: " << rank << " error: " << error << " iter " << it << std::endl;
                if (error < tol) {
                    break;
                }
        }

        assemble_matrix(rank);
        
        if (rank == 0)
            postprocess("approx.vtk", U, n -1 , n - 1, h, h);


        MPI_Finalize();


    }

    void postprocess(const std::string& filename, const std::vector<std::vector<double>>& U, int nx, int ny, double dx, double dy) {
        
        generateVTKFile(filename, U, nx, ny, dx, dy);
    }

    void assemble_matrix(int rank){
        int start_pos = (rank == 0) ? 1 : rank * rows_per_proc;

        std::vector<double> U_flat(rows_per_proc * n);
        for(int i = 0; i < rows_per_proc; i++) {
            for(int j = 0; j < n; j++) {
                U_flat[i * n + j] = U[i + start_pos][j];
            }
        }

        std::vector<double> U_final_flat;

        if (rank == 0) {
            U_final_flat.resize(n * n);
        }

        MPI_Gather(U_flat.data(), rows_per_proc * n, MPI_DOUBLE, U_final_flat.data(), rows_per_proc * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    U[i][j] = U_final_flat[i * n + j];
                }
            }
        }
    }
};