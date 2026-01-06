# From Draft to Official

The main axis of development is through the `RustView` module, with its implementation using the `templated` macro to cover many cases. On the other hand, the `SharedArray` module seems too complex and entangled to be interesting to keep. Some of its features, however, can be worth implementing in `RustView`.

## 1. Components to Keep

The core of Krokkos is around the `RustView` module/struct. This has the most promising results and is the most user-friendly approach for **Rust-Kokkos** interoperability.

*   **`src/rust_view/`**: Core module of the project.
    *   Keep the easy conversion between Rust `ndarray` views/layouts and Kokkos `View`s.
    *   Keep `layout.rs`, `memory_space.rs`, `data_type.rs`, and `dim.rs` to maintain that strongly typed API for better/safer usage.
*   **C++ Backend (`src/cpp/rust_view.cpp`, `src/include/rust_view.hpp`)**: To be kept and cleaned up.
*   **`templated_parser` & `templated_macro`**: Necessary for generating the FFI bindings for `RustView`. Could even use it more.
*   **`RustView` tests**: They cover a good part of the code and should be used in the CI.
*   **`SharedArray` non-owning approach**: If we want to implement a non-owning struct for this interop, the best lead right now is to, as with `shared_array`, have a struct owning a pointer and handing the memory management to the user. This can be done by reusing `shared_array`’s past implementation and removing the `Drop` trait.

## 2. Components to Remove

The current implementation of `SharedArray` as an owning struct is not functional, as it requires a `Drop` trait which can't be implemented. However, some of its past features for non-owning purposes can be reused.

*   **`src/shared_array/`**: Remove entire module.
*   **`src/cpp/shared_array.cpp` & `src/include/shared_array.hpp`**: Remove the corresponding C++ implementations.
*   **`functions_shared_array`**: Remove function bindings.
*   **`KokkosHandles`**: If keeping just one module (`RustView`), can remove `KokkosHandles` for a simpler architecture.

## 3. Improvements & Changes

### Compilation & Build System

The current build process involving `cmake` within `build.rs` has complications when trying to use it with user-written files.

*   **Build Script (`build.rs`)**:
    *   Simplify the build script to link against the shared library.
    *   Remove the `SharedArray` linking directives.
    *   Ensure `cxx` bridges are correctly pointed to the new structure.
*   **User files**: Make it simpler for the user to write their own kernel and use it with Krokkos. This requires a more flexible `build.rs` that can take some user files (C++) to link against.
*   **Full implementation of `RustView`**: Right now, `RustView` only has:
    *   Implementation for 3 dimensions max (need 7).
    *   Implementation for `LayoutRight` and `LayoutLeft` (barely). Need better implementation AND `LayoutStride`.
    *   Implementation for `f64`, `f32`, and `i32` (need more types).
    *   `HostSpace` and `DeviceSpace`. No current support for shared memory spaces.
*   **RustView C++ Implementation**: Some features are functional in the draft but require some more complex patterns to be implemented realisticly (covering the templates of Kokkos::View) like :
    *   DeepCopy,
    *   MirrorView,
    *   SubView,

*   **Restrict views types by user**: Allow for the user to be able to restrict the declinations of the Kokkos::View to be used (restrict for 1D/2D/3D... only, LayoutRight only on hostspace/devicescpae, only one execution space, small set of data types...). This can be done either with Rust's features, or maybe with an initialization function.

*   **More tests, covering more scenarios**.
