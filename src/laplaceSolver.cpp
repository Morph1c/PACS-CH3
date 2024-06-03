#include "../include/laplaceSolver.hpp"
#include "../include/writeVTK.hpp"


namespace parallel_solver{

double LaplaceSolver::local_solver_iter(int start_pos, int end_pos){
    double norm = 0.0;
    
    //#pragma omp parallel for reduction(+:norm) collapse(2)
    for (int i = start_pos; i < end_pos; ++i) {
             // every rank modifies the first row, except the first rank
                                              // since in that case are boundary conditions
                                              // moreover each rank doesn't modify the last row
            for (int j = 1; j < n - 1; ++j) {
                double temp = U(i, j);
                U(i, j) = 0.25 * (U(i-1, j) + U(i+1, j) + U(i, j-1) + U(i, j+1) + h*h*f(i*h, j*h));
                norm += (U(i, j) - temp) * (U(i, j) - temp);
            }
    }
    
    return norm;
}


double LaplaceSolver::parallel_iter(int rank, int size){
        // The code for one iteration of Jacobi goes here

        rows_per_proc = n / size;

        if (n % size != 0) {
            std::cerr << "The number of rows must be divisible by the number of processes" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        } 
    
        double res_k;
        double norm;
        int start_pos = (rank == 0) ? 1 : rank * rows_per_proc;
        int end_pos = (rank == size - 1) ? n - 1 : (rank + 1) * rows_per_proc;


        res_k = 0.0;
        norm = local_solver_iter(start_pos, end_pos);
        
        // Send the last row to the next rank
        // Send the last row to the next rank and receive the first row from the next rank
        MPI_Request requests[4];
        int request_count = 0;

        if (rank < size - 1) {
            MPI_Isend(&U((rank + 1) * rows_per_proc - 1, 0), n, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &requests[request_count++]);
            MPI_Irecv(&U((rank + 1) * rows_per_proc, 0), n, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD, &requests[request_count++]);
        }
        

     // Receive the last row from the previous rank and send the first row to the previous rank
        if (rank > 0) {
            MPI_Isend(&U(rank * rows_per_proc, 0), n, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, &requests[request_count++]);
            MPI_Irecv(&U(rank * rows_per_proc - 1, 0), n, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &requests[request_count++]);
        }

        //MPI_Waitall(request_count, requests, MPI_STATUSES_IGNORE);

        MPI_Barrier(MPI_COMM_WORLD);  // Add this line
        MPI_Allreduce(&norm, &res_k, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    
        return res_k;

}

void LaplaceSolver::solve(int argc, char** argv) {
        // The code for solving the problem goes here
         // Set boundary conditions
        MPI_Init(&argc, &argv);


        int size, rank;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        for (int i = 0; i < n; ++i) {
            U(0, i) = U(i, 0) = U(n-1, i) = U(i, n-1) = 0.0;
            //U_exact[0][i] = U_exact[i][0] = U_exact[n-1][i] = U_exact[i][n-1] = 0.0;
        }

        double error;
        for(int it = 0; it < it_max; ++it){
                error = parallel_iter(rank, size);
                if (sqrt(h*error) < tol) {
                    break;
                }
        }

        assemble_matrix(rank);
        
        if (rank == 0)
            postprocess("approx.vtk", U, n -1 , n - 1, h, h);

        MPI_Finalize();

}

void LaplaceSolver::postprocess(const std::string& filename, 
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& U, int nx, int ny, double dx, double dy) {
        // Add error calculation and VTK output
        generateVTKFile(filename, U, nx, ny, dx, dy);
}

void LaplaceSolver::assemble_matrix(int rank){
        int start_pos = (rank == 0) ? 1 : rank * rows_per_proc;

        std::vector<double> U_flat(rows_per_proc * n);
        for(int i = 0; i < rows_per_proc; i++) {
            for(int j = 0; j < n; j++) {
                U_flat[i * n + j] = U(i + start_pos, j);
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
                    U(i, j) = U_final_flat[i * n + j];
                }
            }
        }
}

}

