#pragma once
#include "rust/cxx.h"
#include <Kokkos_Core.hpp>
#include <string>
#include <iostream>
// #include "KokkosBlas1_nrm1.hpp"

namespace test {
    namespace kernels {

        struct RustViewWrapper;

        struct ViewWrapper {
            virtual void fill(const double* data) = 0;
            virtual size_t size() = 0;
            virtual double get(size_t index) = 0;
            // virtual ~ViewWrapper() = default;
        };


        struct HostView : public ViewWrapper{
            Kokkos::View<double*, Kokkos::DefaultHostExecutionSpace> view;

            HostView(size_t size) :  view("wrappedHostView", size) {}

            void fill(const double* data) override {
                Kokkos::parallel_for("InitView", view.extent(0), KOKKOS_LAMBDA (int i) {
                    view(i) = data[i]; 
                });
            }

            size_t size() override {
                return view.extent(0);
            }

            double get(size_t index) override {
                return view(index); 
            }

        };

        struct DeviceView : public ViewWrapper{
            Kokkos::View<double*, Kokkos::DefaultExecutionSpace> view;

            DeviceView(size_t size) :  view("wrappedDefaultView", size) {}

            void fill(const double* data) override {
                //mirror view and deep copy to access the view stored on device.
                auto h_view = Kokkos::create_mirror_view(view);
                Kokkos::deep_copy(h_view, view);

                Kokkos::parallel_for("InitView", h_view.extent(0), KOKKOS_LAMBDA (int i) {
                    h_view(i) = data[i]; 
                });

                Kokkos::deep_copy(view, h_view);
            }

            size_t size() override {
                return view.extent(0);
            }

            double get(size_t index) override {
                auto host_view = Kokkos::create_mirror_view(view);
                return host_view(index);
            }

        };

        void kokkos_initialize();
        void kokkos_finalize();

        int kernel_mult(/*ViewWrapper a, ViewWrapper b*/);

        RustViewWrapper create_host_view(size_t size);
        RustViewWrapper create_device_view(size_t size);
        void fill_view(const RustViewWrapper& view, rust::Slice<const double> data);
        void show_view(const RustViewWrapper& view);
        void show_execSpace();
        void assert_equal(const RustViewWrapper& view, rust::Slice<const double> data);
    } 
} 