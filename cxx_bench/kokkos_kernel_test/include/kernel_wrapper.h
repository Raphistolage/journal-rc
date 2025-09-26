#pragma once
#include "rust/cxx.h"
#include <Kokkos_Core.hpp>
#include <string>
#include <iostream>
// #include "KokkosBlas1_nrm1.hpp"

namespace test {
    namespace kernels {

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
                Kokkos::parallel_for("InitView", view.extent(0), KOKKOS_LAMBDA (int i) {
                    view(i) = data[i]; 
                });
            }

            size_t size() override {
                return view.extent(0);
            }

            double get(size_t index) override {
                auto host_view = Kokkos::create_mirror_view(view);
                return host_view(index);
            }

        };

        int kernel_mult(/*ViewWrapper a, ViewWrapper b*/);

        std::unique_ptr<HostView> create_host_view(size_t size);
        void fill_view(std::unique_ptr<ViewWrapper>, rust::Slice<const double> data);
        void show_view(std::unique_ptr<ViewWrapper>);
    } 
} 