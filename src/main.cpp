#include <mpi.h>
#include <vector>
#include <cmath>
#include <omp.h>
#include <iostream>
#include "../include/writeVTK.hpp"



int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int n = atoi(argv[1]); // Number of grid points
    double h = 1.0 / (n - 1); // Grid spacing
    auto f = [](double x, double y) { return 8 * M_PI * M_PI * sin(2 * M_PI * x) * sin(2 * M_PI * y); };
    auto u_ex_fun = [](double x, double y) { return sin(2 * M_PI * x) * sin(2 * M_PI * y); };

    int rows_per_proc;

    rows_per_proc = n / size; // Number of rows per process
    
    if (n % size != 0) {
        std::cerr << "The number of rows must be divisible by the number of processes" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }   

    std::vector<std::vector<double>> U(n, std::vector<double>(n, 0.0)); // Initialize U with zeros

    std::vector<std::vector<double>> U_exact(n, std::vector<double>(n, 0.0)); // Initialize U with zeros

    // Set boundary conditions
    for (int i = 0; i < n; ++i) {
            U[0][i] = U[i][0] = U[n-1][i] = U[i][n-1] = 0.0;
            U_exact[0][i] = U_exact[i][0] = U_exact[n-1][i] = U_exact[i][n-1] = 0.0;
    }
    
    

    double norm;
    double res_k;
    size_t it_max = 1000;
    size_t c = 0;
    double tol = 1e-5;

    int start_pos = (rank == 0) ? 1 : rank * rows_per_proc;
    int end_pos = (rank == size - 1) ? n - 1 : (rank + 1) * rows_per_proc;

    for(size_t it = 0; it < it_max; ++it){

        norm = 0.0;
        res_k = 0.0;

        for (int i = start_pos; i < end_pos; ++i) {
             // every rank modifies the first row, except the first rank
                                              // since in that case are boundary conditions
                                              // moreover each rank doesn't modify the last row
            //std::cout << "Rank: " << rank << " i: " << i << std::endl;
            for (int j = 1; j < n - 1; ++j) {
                double temp = U[i][j];
                U[i][j] = 0.25 * (U[i-1][j] + U[i+1][j] + U[i][j-1] + U[i][j+1] + h*h*f(i*h, j*h));
                norm += (U[i][j] - temp) * (U[i][j] - temp);
            }
            //std::cout << "Rank: " << rank << " norm: " << norm << " iter " << it << " i " << i << std::endl;

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

        if (std::sqrt(h*res_k) < tol) {
            break;
        }
        c++;
    }
    std::cout << "Finished, rank " << rank << " res " << res_k << " iters: " << c << std::endl;

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

        generateVTKFile("laplace_approx.vtk", U, n - 1, n - 1, h, h);
        generateVTKFile("laplace_exact.vtk", U_exact, n - 1, n - 1, h, h);
    }

    
    MPI_Finalize();

    
    return 0;

}

