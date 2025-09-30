# Rust ↔ C++ Interoperability

This repository contains examples exploring different approaches to Rust-C++ interoperability, with focus on high-performance computing libraries.

## Overview

This small project investigates various patterns and techniques for bridging Rust and C++ code, particularly for:
- Linear algebra libraries (Armadillo)
- Parallel computing frameworks (Kokkos)
- Memory management strategies
- Performance considerations

## Examples
- **[Armadillo Basic](examples/armadillo-basic/)** - Simple linear algebra operations using Armadillo
- **[Kokkos Advanced](examples/kokkos-advanced/)** - Parallel computing with Kokkos views and kernels
- **[Simple Wrapper](examples/simple-wrapper/)** - Basic C wrapper patterns and techniques

### Structure
```
examples/          # Working examples
benchmarks/        # Performance comparisons
docs/              # Documentation and daily notes
tools/             # Build scripts and utilities  
```

## Quick Start

Each example stands on it's own:

```bash
cd examples/kokkos-advanced
cargo build
cargo run
```

## Requirements

- Rust (latest stable)
- C++ compiler with C++17/20 support
- CMake
- Kokkos (for kokkos-advanced example)
- Armadillo (for armadillo-basic example)

## Structure

- [Getting Started](docs/getting-started.md) - Setup and usage
- [Cxx for Rust](docs/Cxx%20for%20Rust-Kokkos.md) - Tool choice
- [Performance Analysis](docs/benchmarks.md) - Benchmark results
- [Daily Notes](docs/daily-notes/) - Journal and observations
