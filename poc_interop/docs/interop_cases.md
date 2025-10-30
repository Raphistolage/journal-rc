- [x] Instantiate Rust variable (that implements toShared) --> cast to SharedArrayView --> call C++ function on it --> cast it into Kokkos::mdspan on Host --> call Kokkos kernel on it (ExecSpace = HostSpace) --> cast back to Rust Variable type (fromShared) 

https://github.com/Raphistolage/journal-rc/blob/26264bf110dc6bf9d84540887db09621599c127d/poc_interop/src/mdspan_interop/ops.rs#L128-L159
&& 
https://github.com/Raphistolage/journal-rc/blob/26264bf110dc6bf9d84540887db09621599c127d/poc_interop/src/mdspan_interop/ops.rs#L43-L58

- [ ] Instantiate Rust variable (that implements toShared) --> cast to SharedArrayView --> move the data to device --> call C++ function on it --> cast it into Kokkos::mdspan on Device --> call Kokkos kernel on it (ExecSpace = Cuda,HIP...) --> copy to host --> cast back to Rust Variable type (fromShared) 

- [x] From Rust instantiate Kokkos::View (call to constructor) --> receive OpaqueView {
        view: UniquePtr<IView>,

        size: u32,

        rank: u32,

        shape: Vec<i32>,

        mem_space: MemSpace,

        layout: Layout,
} --> call Kokkos kernel on it --> cast opaqueView to mdspan (host) or kokkos::view (device) & execute kernel --> receive result (opaqueView or other).

https://github.com/Raphistolage/journal-rc/blob/dcd62f3567393295505951d6fb10f4ed56738e71/poc_interop/src/rust_view/ops.rs#L56-L76
&&
https://github.com/Raphistolage/journal-rc/blob/dcd62f3567393295505951d6fb10f4ed56738e71/poc_interop/src/cpp/view_wrapper.cpp#L312-L341

- [ ] Write a C++ function using kernels and views : void func(view1, view2) {Kokkos::parallel_for(... {})} --> call it from fn rust_func(array1, array2) --> array1.toShared --> ffi::call_kokkos_func(shared_array1, shared_array2) --> view1 = shared_array1.from_shared() --> func(view1, view2)                (Link with rust macro ?).

- [ ] Instantiate Kokkos::mdspan on C++ side --> cast to SharedArrayView (toShared) --> call Rust function on it --> cast it into ndarray (or anything that implements fromShared) --> execute Rust function on it --> cast back to Kokkos::mdspan (fromShared).

https://github.com/Raphistolage/journal-rc/blob/26264bf110dc6bf9d84540887db09621599c127d/poc_interop/src/include/mdspan_interop.hpp#L98-L107

- [ ] From C++ instantiate ndarray (call to constructor) --> receive OpaqueArray --> call Rust function on it --> cast opaqueArray & execute Rust function --> access modified OpaqueArray from C++ (extract data with built-in function calling Rust side)


