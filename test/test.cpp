#include <iostream>
#include <chrono>
#include <cmath>
#include <fstream>
#include "../src/laplaceSolver.hpp"



using namespace parallel_solver;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    double tol = 1e-5;
    auto f = [](double x, double y) { return 8 * M_PI * M_PI * sin(2 * M_PI * x) * sin(2 * M_PI * y); };
    auto g = [](double x, double y) { return 0 ;};
    auto u_ex_fun = [](double x, double y) { return sin(2 * M_PI * x) * sin(2 * M_PI * y); };
    std::vector<double> times;
    std::vector<double> ns;

    std::ofstream file("../plots/times.txt");
    for (int k = 4; k <= 8; k++){
        int n = pow(2, k);
        LaplaceSolver solver(n, tol, f, g, u_ex_fun);
        solver.solve();

        double local_time = solver.get_execution_time();
        double global_time;

        MPI_Reduce(&local_time, &global_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        
        int world_size;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        global_time /= world_size;

        if (global_time != 0){
            file << n << " " << global_time << "\n";
        }        
    }

    MPI_Finalize();
    file.close();
    
    return 0;
}
