# Kokkos-Rust Integration

Example demonstrating Kokkos parallel computing integration with Rust through CXX.

## Purpose

- **Kokkos View Wrapping** - Safe abstraction over Kokkos::View with CXX
- **Execution Space Management** - Support for Host and Device execution spaces
- **Memory Management** - Proper lifecycle management for Kokkos resources
- **Polymorphic Design** - Base class abstraction with concrete implementations

## Architecture

```
RustViewWrapper
├── ViewWrapper (abstract base)
│   ├── HostView (Kokkos::HostSpace)
│   └── DeviceView (Kokkos::DefaultExecutionSpace)
└── ExecSpace (execution context)
```

## Requirements

- Kokkos installed in `/usr/local` or `~/kokkos-install`
- OpenMP support
- C++20 compiler
- OpenBLAS and LAPACK

## Key Features

### Memory Safety
- Automatic Kokkos initialization/finalization
- Safe downcasting with execution space tracking

### Performance
- Zero-copy data access where possible
- Efficient host-device memory transfers
- Use of Kokkos' parallel optimization

### Flexibility
- Support for multiple execution spaces
- Extensible design for new view types
- Template-based implementation patterns

## Overview

```rust
unsafe {
    // Initialize Kokkos
    ffi::kokkos_initialize();
    
    // Create a view on the host
    let view = ffi::create_host_view(1000);
    
    // Fill with data
    let data = vec![1.0; 1000];
    ffi::fill_view(&view, &data);
    
    // Clean up
    ffi::kokkos_finalize();
}
```

## Issues

- Instantiation complexity with multiple template to adapt into parameters/struct