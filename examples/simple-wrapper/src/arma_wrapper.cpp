#include "arma_wrapper.h"
#include <armadillo>
#include <iostream>

extern "C" {

void arma_matmul(const double* a, const double* b, double* c, int m, int k, int n) {
    arma::Mat<double> A(const_cast<double*>(a), m, k, false, true);
    arma::Mat<double> B(const_cast<double*>(b), k, n, false, true);

    // std::cout << "Matrix A (" << m << "x" << k << "):\n" << A << std::endl;
    // std::cout << "Matrix B (" << k << "x" << n << "):\n" << B << std::endl;

    arma::Mat<double> C(A * B);
    std::memcpy(c, C.memptr(), sizeof(double) * m * n);
}

void raise_mat(double* a, int len, double k) {
    for (int i = 0; i < len; ++i) {
        a[i] += k;
    }
}

}