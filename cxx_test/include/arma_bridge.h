#pragma once
#include "rust/cxx.h"
#include <armadillo>

namespace org {
    namespace armadillo {

        struct Mat {
            arma::Mat<double> mat;
        };

        struct RustMat;

        void arma_matmul(const double* a, const double* b, double* c, int m, int k, int n);
        // void transpose(Mat& a);
        // double* mat_data(const std::unique_ptr<Mat>& a);
        // rust::Box<RustMat> transpose_and_raise(Mat& a);
        void raise_mat(double* a, int len, double k);


    } // namespace armadillo
} // namespace org