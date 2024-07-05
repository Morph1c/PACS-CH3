
// clang-format off

//@note Dont put the full path in the include directive, just the name of the file
// #include "laplaceSolver.cpp"
// #include "writeVTK.cpp"  
// and then you let the compiler to find where you store the header files
// Moreover, header files are usually stored in the include directory
#include "../src/laplaceSolver.hpp" //@note moreover this is the directori wher also this file lives!
#include "../include/writeVTK.hpp"


namespace parallel_solver{


/* @note
This is a really just a detail, but the methods solve() local_solver_iter and parallel)iter()  are "morally const" in the sense that 
you expect to be able to run them also on const LaplaceSolver objects.
They modify some private members of the class, but thos members are just implementation details, not really part
of the class state. So, it is more appropriate to declare this methods as const, and the private variable used to cache 
results as mutable.
*/
double LaplaceSolver::local_solver_iter(int rank){
    double norm = 0.0;
    double temp = 0.0;
    // depending on the architecture this directive can be slow a lot the execution
    // so i decided to comment it, but if one want to abilitate hybrid programming leave the comment
    //#pragma omp parallel for reduction(+:norm) shared(local_U, temp) collapse(2)
        for (int i = 1; i < local_U.rows() - 1; ++i) {
             // every rank modifies the first row, except the first rank
                                              // since in that case are boundary conditions
                                              // moreover each rank doesn't modify the last row
            //@note  You have a data race here, because two threads may be reading and writing to the same memory location
            for (int j = 1; j < n - 1; ++j) {
                temp = local_U(i, j);
                local_U(i, j) = 0.25 * (local_U(i-1, j) + local_U(i+1, j) + local_U(i, j-1) + local_U(i, j+1) + h*h*local_F(i, j));
                norm += (local_U(i, j) - temp) * (local_U(i, j) - temp);
            }
        }
    

  
    return sqrt(h*norm);
}


double LaplaceSolver::parallel_iter(int rank, int size){
        // The code for one iteration of Jacobi goes here    
        double error;
     
        error = local_solver_iter(rank);
        
        // Send the last row to the next rank
        // Send the last row to the next rank and receive the first row from the next rank
        if (rank < size - 1) {
            MPI_Request send_request;
            MPI_Isend(local_U.row(local_U.rows() - 2).data(), n, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &send_request);
                
            Eigen::VectorXd new_last_row(n);
            MPI_Request recv_request;
            MPI_Irecv(new_last_row.data(), n, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &recv_request);
                
            MPI_Wait(&recv_request, MPI_STATUS_IGNORE);
            local_U.row(local_U.rows() - 1) = new_last_row;
                
            MPI_Wait(&send_request, MPI_STATUS_IGNORE);
        }
        

     // Receive the last row from the previous rank and send the first row to the previous rank
        if (rank > 0) {
            Eigen::VectorXd new_first_row(n);
            MPI_Request recv_request;
            MPI_Irecv(new_first_row.data(), n, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &recv_request);
            
            MPI_Wait(&recv_request, MPI_STATUS_IGNORE);
            local_U.row(0) = new_first_row;

            MPI_Request send_request;
            MPI_Isend(local_U.row(1).data(), n, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &send_request);
            
            MPI_Wait(&send_request, MPI_STATUS_IGNORE);
    
        }

        //MPI_Waitall(request_count, requests, MPI_STATUSES_IGNORE);

        MPI_Barrier(MPI_COMM_WORLD);  // Add this line
        //MPI_Allreduce(&norm, &res_k, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); do this if you want to compute average residual

        return error;

}

void LaplaceSolver::solve(){
        // I decide to inizialie the MPI communication in the main function
        // so that i can test multiple solvers in the same main function        
        auto start = std::chrono::high_resolution_clock::now();

        int size, rank;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        rows_per_proc = n / size; 
        rows_per_rank = std::vector<int>(size, rows_per_proc);
        real_start_pos = std::vector<int>(size, 0.0);

        // then add 1 to each rank until reach n
        for (int i = 0; i < n % size; i++) {
            rows_per_rank[i] = rows_per_proc + 1;
        }

        for(int i = 1; i < size; i++) {
            real_start_pos[i] = real_start_pos[i - 1] + rows_per_rank[i - 1];
        }

        // divide the matrix
        assemble_local_matrix(rank, size);
        assemble_local_F(rank, size);

        double error;
        int it = 0;
        int converged_res = 0;
        do{
                error = parallel_iter(rank, size);
                int res_k_true = (error < tol) ? 1 : 0;
                MPI_Allreduce(&res_k_true, &converged_res, 1, MPI_INT, MPI_PROD, MPI_COMM_WORLD); //if all the processes have converged, the product will be 1 --> we stop
                it++;
        }while(it < it_max and !converged_res);

        assemble_matrix(rank, size);
        
        if (rank == 0){
            if(it >= it_max){
                std::cout << "The solver did not converge in " << it << " iterations" << std::endl;
            }
            else{
                std::cout << "The solver converged in " << it << " iterations" << std::endl;
            }
            postprocess("../plots/approx.vtk", U, n - 1 , n - 1, h, h);
            postprocess("../plots/error.vtk", Error_field, n - 1, n - 1, h, h);
        }

        auto end = std::chrono::high_resolution_clock::now();
        diff = end - start;
}

void LaplaceSolver::postprocess(const std::string& filename, 
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& U, int nx, int ny, double dx, double dy) {
        // Add error calculation and VTK output
        generateVTKFile(filename, U, nx, ny, dx, dy);
}

void LaplaceSolver::assemble_local_matrix(int rank, int size){
    // allocate memory for local_U
    if (rank == 0 or rank == size - 1)
        local_U = Eigen::MatrixXd::Zero(rows_per_rank[rank] + 1, n);
    else
        local_U = Eigen::MatrixXd::Zero(rows_per_rank[rank] + 2, n);
    
    // set the boundary conditions
    boundary_conditions(rank, size);
}

void LaplaceSolver::assemble_local_F(int rank, int size){
    // allocate memory for local_F
    if (rank == 0 or rank == size - 1)
        local_F = Eigen::MatrixXd::Zero(rows_per_rank[rank] + 1, n);
    else
        local_F = Eigen::MatrixXd::Zero(rows_per_rank[rank] + 2, n);

    for(int i = 1; i < local_F.rows() - 1; i++) {
        for(int j = 1; j < n - 1; j++) {
            if(rank == 0){
                local_F(i, j) = f((real_start_pos[rank] + i) * h, j * h);
            }
            else{
                local_F(i, j) = f((real_start_pos[rank] + i - 1) * h, j * h);
            }

        }
    }
}

void LaplaceSolver::assemble_matrix(int rank, int size){
        int start_pos = (rank == 0) ? 0 : 1;
        int end_pos = (rank == size - 1) ? local_U.rows() : local_U.rows() - 1;

        std::vector<double> U_flat(rows_per_rank[rank] * n);
        for(int i = start_pos; i < end_pos; i++) {
            for(int j = 0; j < n; j++) {
                U_flat[(i - start_pos) * n + j] = local_U(i, j);
            }
        }

        std::vector<double> U_final_flat;
        std::vector<int> recvcounts;
        std::vector<int> displs;

        if (rank == 0) {
            U_final_flat.resize(n * n);
            recvcounts.resize(size);
            displs.resize(size);
            for (int i = 0; i < size; i++) {
                recvcounts[i] = rows_per_rank[i] * n;
                displs[i] = (i > 0) ? displs[i - 1] + rows_per_rank[i - 1] * n: 0;
                //displs[i] = (i > 0) ? (displs[i - 1] + recvcounts[i - 1]) : 0;
            }
        }

        MPI_Gatherv(U_flat.data(), rows_per_rank[rank] * n, MPI_DOUBLE, U_final_flat.data(), recvcounts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0) {

            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    U(i, j) = U_final_flat[i * n + j];
                    Error_field(i, j) = U(i, j) - u_ex_fun(i * h, j * h);
                }
            }
        }
}

void LaplaceSolver::boundary_conditions(int rank, int size){
    // set val to get homogenous Dirichlet boundary conditions with given constant
    // easly generalized to non-constant values
    if (rank == 0){
        for (int i = 0; i < n; i++){
            local_U(0, i) = g(0, i*h);
        }
    }
    else if (rank == size - 1){
        for (int i = 0; i < n; i++){
            local_U(local_U.rows() - 1, i) = g(1, i*h); // considering [0, 1]^2 domain
        }
    }

    for(int i = 1; i < local_U.rows() - 1; i++){
            local_U(i, 0) = g((real_start_pos[rank] + i - 1) * h, 0);
            local_U(i, n - 1) = g((real_start_pos[rank] + i - 1) * h, 1); // consider [0, 1]^2 domain
    }
    
}

double LaplaceSolver::get_execution_time(){
    return diff.count();
}

}

