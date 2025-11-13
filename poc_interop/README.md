# Rust ↔ C++ Interoperability

This repository contains examples exploring different approaches to Rust-Kokkos interoperability, in particular how to manipulate Kokkos::view from Rust and pass it to kernels.

The most practical implementation for now is in the rust_view module, with the RustView struct.

## Requirements

- Rust (latest stable)
- C++ compiler with C++17/20 support
- CMake
