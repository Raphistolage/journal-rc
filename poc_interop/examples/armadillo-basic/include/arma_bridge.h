#pragma once
#include "rust/cxx.h"
#include <armadillo>

namespace org {
    namespace armadillo {

        // struct Mat {
        //     arma::Mat<double> mat;
        // };

        struct RustMat;

        void arma_matmul(const RustMat& a, const RustMat& b, RustMat& c);
        // void transpose(Mat& a);
        // double* mat_data(const std::unique_ptr<Mat>& a); 
        void transpose_and_raise(RustMat &a);
        //void raise_mat(double* a, int len, double k);


    } // namespace armadillo
} // namespace org