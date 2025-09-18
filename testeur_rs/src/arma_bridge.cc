#include <armadillo>
#include <iostream>

#include "testeur_rs/include/arma_bridge.h"
#include "testeur_rs/src/main.rs.h"

namespace org {
namespace armadillo {
// Multiplies two matrices (A: m x k, B: k x n), result in C (m x n)

// Mat donne une matrice par ordre column (en layout left en fait)

void arma_matmul(const double* a, const double* b, double* c, int m, int k, int n) {
    arma::Mat<double> A(const_cast<double*>(a), m, k, true, true);
    arma::Mat<double> B(const_cast<double*>(b), k, n, true, true);

    std::cout << "Matrix A (" << m << "x" << k << "):\n" << A << std::endl;
    std::cout << "Matrix B (" << k << "x" << n << "):\n" << B << std::endl;


    arma::Mat<double> C = (A * B).as_row();
    std::memcpy(c, C.memptr(), sizeof(double) * m * n);
    // for (int i = 0; i < m; ++i) {
    //     for (int j = 0; j < n; ++j) {
    //         c[i * n + j] = 0.0;
    //         for (int l = 0; l < k; ++l) {
    //             c[i * n + j] += a[i * k + l] * b[l * n + j];
    //         }
    //     }
    // }
}

}
}
