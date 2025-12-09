# Rust ↔ Kokkos Interoperability Proof of Concept
[![Tests](https://github.com/Raphistolage/journal-rc/actions/workflows/tests.yml/badge.svg)](https://github.com/Raphistolage/journal-rc/actions/workflows/tests.yml)

This repository aims to explore different approaches to Rust-Kokkos interoperability, in particular how to manipulate Kokkos::view from Rust and pass it to kernels.

The most practical implementation for now is in the rust_view module, with the RustView struct.

There is also a procedural macro attribute "template" that allows for fairly easy binding with templated functions from C++.

# Quickstart

You can use the kokkos-install.sh script to install Kokkos.

TODO : Proper setup guide

# Content

**poc_interop**
has all the different approach to this problem :
- **OpaqueView** : An opaque type (opaque to Rust) that can only be created by a call to a C++ constructor with data from Rust caller (data: impl Into<Vec<f64>>>), that holds a **Kokkos::View** (unmanaged).
- **SharedArray** : A **non-owning** struct shared through Cxx bridge. This aims to be a lightweight wrapper around the data raw pointer that can be easily casted **to** and **from** a **Kokkos::View**.
- **RustView** : A lightweight, strongly typed, wrapper around **OpaqueView** that enables idiomatic and safe Rust manipulation of the Kokkos::View.

**Templated_macro** is a procedural macro attribute that generates a **cxx bridge** to link a **templated C++ function** to strongly typed **rust functions.**

**Templated_parser** is the **parser** that need to be **called in the build.rs file** of your project to **generate** the bridge and **link** it. It's usage is very similar to Cxx's.

# Requirements

- Rust (latest stable)
- C++ compiler with C++17/20 support
- CMake
