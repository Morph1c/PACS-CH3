#include <mpi.h>
#include <vector>
#include <cmath>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int n = atoi(argv[1]); // Number of grid points
    double f = atof(argv[2]); // Forzant for the Laplace equation

    int rows_per_proc = n / size; // Number of rows per process

    std::vector<std::vector<double>> U(n, std::vector<double>(n, 0.0)); // Initialize U with zeros

    // Set boundary conditions
    if (rank == 0) {
        for (int i = 0; i < n; ++i) {
            U[0][i] = U[i][0] = U[n-1][i] = U[i][n-1] = 1.0;
        }
    }

    double norm;
    do {
        norm = 0.0;
        for (int i = rank * rows_per_proc + 1; i < (rank + 1) * rows_per_proc - 1; ++i) {
            for (int j = 1; j < n - 1; ++j) {
                double temp = U[i][j];
                U[i][j] = 0.25 * (U[i-1][j] + U[i+1][j] + U[i][j-1] + U[i][j+1] - f);
                norm += (U[i][j] - temp) * (U[i][j] - temp);
            }
        }

        // Communicate boundary rows
        if (rank > 0) {
            MPI_Send(&U[rank * rows_per_proc][0], n, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD);
        }
        if (rank < size - 1) {
            MPI_Recv(&U[(rank + 1) * rows_per_proc - 1][0], n, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (rank < size - 1) {
            MPI_Send(&U[(rank + 1) * rows_per_proc - 2][0], n, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD);
        }
        if (rank > 0) {
            MPI_Recv(&U[rank * rows_per_proc - 1][0], n, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        MPI_Allreduce(MPI_IN_PLACE, &norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        norm = sqrt(norm);
    } while (norm > 1e-5);

    MPI_Finalize();
    return 0;

}

// prompt was:
// Write a parallel c++ code using MPI for solving the laplace equation on a square w x h. 
// We aim at representing the solution as a (dense) matrix U of size n × n; 
// the matrix is initialized with zeroes, except for the first and last rows and columns, 
// which contain the boundary condition values. 
// The algorithm consists of the following iterative procedure. 
// 1) update each internal entries as the average of the values of a four–point stencil 
// 2)compute the convergence criterion as the norm of the increment between U^(k+1) and U^(k) 
// 3) repeat until convergence criterion is reached. 
// In particular: 
// • The number n(points of the discrete grid), f(is the forzant for the laplace equation) 
// and the number of parallel tasks should be given by the user; 
// • The nodes are split into blocks of rows and assigned to different MPI processes; 
// • The number of rows owned by each processor should be balanced, 
// i.e. it should be as constant as possible among the different MPI ranks;