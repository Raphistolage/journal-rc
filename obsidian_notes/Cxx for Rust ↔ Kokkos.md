
# Context
- FFI interoperability between Rust and Kokkos (C++)

# Pros of Cxx
- Fast  (does not become the bottleneck)
- Scalable (to follow Rust's and Kokkos' updates)
- Bidirectional
- Implemented in Rust,  benefit from borrow checker and ownership system
- Centralize everything in a single dual interface.
- Handles C++ compilation (benefit from static assertions) and linking
- Maintained by reliable developer, used in industry

# Cons of Cxx

- Not all types can be passed directly (only basic ones), thus the need for shared ptr
- Using Kokkos' types and macros is complex
- Need wrapper around the C++ library anyway
- Debugging can be hard to read


