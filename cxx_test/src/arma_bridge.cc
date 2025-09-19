#include <armadillo>
#include <iostream>

#include "testeur_rs/include/arma_bridge.h"
#include "testeur_rs/src/main.rs.h"

namespace org {
    namespace armadillo {

        // Multiplies two matrices (A: m x k, B: k x n), result in C (m x n)

        // Mat donne une matrice par ordre column (en layout left en fait)

        std::unique_ptr<Mat> arma_matmul(const double* a, const double* b, int m, int k, int n) {
            arma::Mat<double> A(const_cast<double*>(a), m, k, true, true);
            arma::Mat<double> B(const_cast<double*>(b), k, n, true, true);

            std::cout << "Matrix A (" << m << "x" << k << "):\n" << A << std::endl;
            std::cout << "Matrix B (" << k << "x" << n << "):\n" << B << std::endl;

            auto C = std::make_unique<Mat>();
            C->mat = (A * B).as_row();

            return C;
        }

        void transpose(Mat& a) {
            a.mat = a.mat.t();
        }

        rust::Box<RustMat> transpose_and_raise(Mat& a) {
            a.mat = a.mat.t();
            auto raisedMat = raise_mat(a.mat.memptr(), static_cast<int>(a.mat.n_elem), 25.0);
            return raisedMat;
        }

        double* mat_data(const std::unique_ptr<Mat>& a) {
            return a->mat.memptr();
        }

        

    }
}
