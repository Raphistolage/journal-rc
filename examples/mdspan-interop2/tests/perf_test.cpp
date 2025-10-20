/*

MADE ENTIRELY BY AI (COPILOT CLAUDE SONNET 4.5) AND IS SOLELY USED FOR TESTING AND PERFS COMPARISON PURPOSE.

*/



#include <iostream>
#include <chrono>
#include <mdspan>
#include <vector>

// Simple 2x2 matrix product using mdspan
std::vector<double> matrix_product(
    std::mdspan<const double, std::dextents<std::size_t, 2>> mat1,
    std::mdspan<const double, std::dextents<std::size_t, 2>> mat2)
{
    if (mat1.extent(1) != mat2.extent(0)) {
        throw std::runtime_error("Incompatible sizes of matrices");
    }

    std::vector<double> result(mat1.extent(0) * mat2.extent(1));

    for (size_t i = 0; i < mat1.extent(0); i++)
    {
        for (size_t j = 0; j < mat2.extent(1); j++)
        {
            double r = 0;
            for (size_t k = 0; k < mat1.extent(1); k++)
            {
                r += mat1[i, k] * mat2[k, j];
            }
            result[i * mat2.extent(1) + j] = r;
        }
    }
    
    return result;
}

int main() {
    std::vector<double> v = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,};
    std::vector<double> s = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,};

    std::cout << "Test Matrix Product using mdspan:" << std::endl;

    double tot_time = 0.0;
    const int N = 1'000'000;



    for (int i = 0; i < N; i++) {
        v[0] += 1.0;
        s[1] += 2.0;

        // Create mdspan views (2x2 matrices)
        std::mdspan<const double, std::dextents<std::size_t, 2>> mat1(v.data(), 4, 30);
        std::mdspan<const double, std::dextents<std::size_t, 2>> mat2(s.data(), 30, 4);

        auto start = std::chrono::high_resolution_clock::now();
        
        auto result = matrix_product(mat1, mat2);
        
        auto end = std::chrono::high_resolution_clock::now();
        
        std::chrono::duration<double> elapsed = end - start;
        tot_time += elapsed.count();
    }

    std::cout << "Total time elapsed in seconds: " << tot_time << std::endl;
    std::cout << "Time elapsed in average per 4x30 matrix product: " << (tot_time / N) << " seconds" << std::endl;
    std::cout << "Average time in nanoseconds: " << (tot_time / N * 1e9) << " ns" << std::endl;

    return 0;
}
