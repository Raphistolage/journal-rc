#include <Kokkos_Core.hpp>
#include <iostream>
#include <mdspan>

#include "kernel_wrapper.h"
#include "rust-kokkos-interop/src/lib.rs.h"


namespace rust_kokkos_interop {
    void kokkos_initialize() {
        if (!Kokkos::is_initialized()) {
            Kokkos::initialize();
            std::cout << "Kokkos initialized successfully!" << std::endl;
        } else {
            std::cout << "Kokkos is already initialized." << std::endl;
        }
    }

    void kokkos_finalize() {
        if (Kokkos::is_initialized()) {
            Kokkos::finalize();
            std::cout << "Kokkos finalized successfully!" << std::endl;
        } else {
            std::cout << "Kokkos is not initialized." << std::endl;
        }
    } 
    
    template <int D>
    std::mdspan<const double, std::dextents<std::size_t, D>> from_shared(SharedArrayView arrayView) {
        const int* shape = arrayView.shape.data();
        switch(arrayView.rank) {
            case 1:
                return std::mdspan<const double, std::dextents<std::size_t, D>>(arrayView.ptr, shape[0]);
                break;
            case 2:
                return std::mdspan<const double, std::dextents<std::size_t, D>>(arrayView.ptr, shape[0], shape[1]);
                break;
            case 3:
                return std::mdspan<const double, std::dextents<std::size_t, D>>(arrayView.ptr, shape[0], shape[1], shape[2]);
                break;
            case 4:
                return std::mdspan<const double, std::dextents<std::size_t, D>>(arrayView.ptr, shape[0], shape[1], shape[2], shape[3]);
                break;
            case 5:
                return std::mdspan<const double, std::dextents<std::size_t, D>>(arrayView.ptr, shape[0], shape[1], shape[2], shape[3], shape[4]);
                break;
            case 6:
                return std::mdspan<const double, std::dextents<std::size_t, D>>(arrayView.ptr, shape[0], shape[1], shape[2], shape[3], shape[4], shape[5]);
                break;
            case 7:
                return std::mdspan<const double, std::dextents<std::size_t, D>>(arrayView.ptr, shape[0], shape[1], shape[2], shape[3], shape[4], shape[5], shape[6]);
                break;
        }
    }

    Errors deep_copy(SharedArrayViewMut& arrayView1, const SharedArrayView& arrayView2) {
        int rank1 = arrayView1.rank;
        int rank2 = arrayView2.rank;
        const int* shape1 = arrayView1.shape.data();
        const int* shape2 = arrayView2.shape.data();
        const int* stride2 = arrayView2.stride.data();
        if (rank1 != rank2){
            std::cout << "Both views should be of same rank. \n Deep copy aborted." << "\n";
            return Errors::IncompatibleRanks;
        }
        switch (rank1)
        {
            case 1: {
                if (shape1[0] != shape2[0])
                {
                    std::cout << "Both views should have same shapes \n Deep copy aborted." << "\n";
                    return Errors::IncompatibleShapes;
                }
                for (int i = 0; i < shape1[0]; i++)
                {
                    arrayView1.ptr[i*stride2[0]] = arrayView2.ptr[i*stride2[0]];
                }
                break;
            }
            case 2: {
                if (shape1[0] != shape2[0] || shape1[1] != shape2[1])
                {
                    std::cout << "Both views should have same shapes \n Deep copy aborted." << "\n";
                    return Errors::IncompatibleShapes;
                }
                for (int i = 0; i < shape1[0]; i++)
                {
                    for (int j = 0; j < shape1[1]; j++)
                    {
                        arrayView1.ptr[i*stride2[0]+j*stride2[1]] = arrayView2.ptr[i*stride2[0]+j*stride2[1]];
                    }
                }
                break;
            }
            case 3: {
                if (shape1[0] != shape2[0] || shape1[1] != shape2[1] || shape1[2] != shape2[2])
                {
                    std::cout << "Both views should have same shapes \n Deep copy aborted." << "\n";
                    return Errors::IncompatibleShapes;
                }
                for (int i = 0; i < shape1[0]; i++)
                {
                    for (int j = 0; j < shape1[1]; j++)
                    {
                        for (int k = 0; k < shape1[2]; k++)
                        {
                            arrayView1.ptr[i*stride2[0]+ j*stride2[1] + k*stride2[2]] = arrayView2.ptr[i*stride2[0]+ j*stride2[1] + k*stride2[2]];
                        }
                    }
                }
                break;
            }
            default:
                break;
        }
        return Errors::NoErrors;
    }

    double dot(SharedArrayView shared_arr1, SharedArrayView shared_arr2) {
        auto arr1 = from_shared(shared_arr1);
        Kokkos::View(arr1);
        auto arr2 = from_shared(shared_arr2);
        Kokkos::View(arr2);
        return 0.0;
    }

}

