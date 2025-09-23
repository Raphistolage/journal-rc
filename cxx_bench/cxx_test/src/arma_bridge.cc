#include <armadillo>
#include <iostream>

#include "testeur_rs/include/arma_bridge.h"
#include "testeur_rs/src/main.rs.h"

namespace org {
    namespace armadillo {

        // Multiplies two matrices (A: m x k, B: k x n), result in C (m x n)

        // Mat donne une matrice par ordre column (en layout left en fait)

        void arma_matmul(const RustMat& a, const RustMat& b, RustMat& c) {
            const rust::Vec<double>* AVec = &a.mat;  // AVec is a pointer
            const double* a_ptr = (*AVec).data();       
            const rust::Vec<double>& BVec = b.mat;
            const double* b_ptr = BVec.data();
            arma::Mat<double> A(const_cast<double*>(a_ptr), a.i, a.j, false, true);
            arma::Mat<double> B(const_cast<double*>(b_ptr), b.i, b.j, false, true);

            // std::cout << "Matrix A (" << a.i << "x" << a.j << "):\n" << A << std::endl;
            // std::cout << "Matrix B (" << b.i << "x" << b.j << "):\n" << B << std::endl;

            arma::Mat<double> C_result(A * B);
            std::memcpy(const_cast<double*>(c.mat.data()), C_result.memptr(), sizeof(double) * a.i * b.j);
        }

        // void transpose(Mat& a) {
        //     a.mat = a.mat.t();
        // }

        void transpose_and_raise(RustMat &a) {        
            // Create Armadillo matrix from RustMat data
            arma::Mat<double> A(const_cast<double*>(a.mat.data()), a.i, a.j, false, true);
            
            // Transpose the matrix
            //arma::Mat<double> transposed = A.t();
            
            // Raise each element by 25.0
            A += 25.0;

            // Copy transposed data back to the original Vec (assumes same total size)
            std::memcpy(const_cast<double*>(a.mat.data()), A.memptr(), sizeof(double) * A.n_elem);
            // Update dimensions
            a.i = a.j; // swap dimensions for transpose
            a.j = A.n_rows; // original rows become new cols
        }

        // double* mat_data(const std::unique_ptr<Mat>& a) {
        //     return a->mat.memptr();
        // }

        // void raise_mat(double* a, int len, double k) {
        //     for (int i = 0; i < len; ++i) {
        //         a[i] += k;
        //     }
        // }

    }
}
