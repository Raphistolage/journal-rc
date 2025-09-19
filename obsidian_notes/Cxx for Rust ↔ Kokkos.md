# Context
- FFI interoperability between Rust and Kokkos (C++)

# Pros of Cxx
- Fast (does not become the bottleneck)
- Scalable (keeps up with Rust’s and Kokkos’ updates)
- Bidirectional
- Implemented in Rust, benefits from the borrow checker and ownership system
- Centralizes everything in a single dual interface
- Handles C++ compilation (benefits from static assertions) and linking
- Maintained by a reliable developer, used in industry

# Cons of Cxx
- Not all types can be passed directly (only basic ones), hence the need for `shared_ptr`
- Using Kokkos’ types and macros is complex
- Requires wrappers around the C++ library anyway
- Debugging output can be hard to read

# Comparison of sizes and performance

![CxxSize](./images/cxx_size.png)  
![WrapperSize](./images/wrapper_size.png)  

![CxxPerf](./images/cxx_test.png)  
![WrapperPerf](./images/wrapper_test.png)  

We see no significant difference in this mini test project, implemented once with an unsafe `extern "C"` wrapper for direct FFI, and once with Cxx.

Most notably, we see that adding Cxx does not slow down the program’s performance on average.
