### Interop cases : 

Instantiate Rust variable (that implements toShared) --> cast it into Kokkos::View --> call Kokkos kernel on it

From Rust instantiate Kokkos::View (call to constructor) --> call Kokkos kernel on it --> receive value into an ndarray (or anything that implements fromShared)

Instantiate Kokkos::View --> cast it into ndarray (or anything that implements fromShared) --> call Rust function on it 

From C++ instantiate ndarray (call to constructor) --> call Rust function on it --> receive value into a Kokkos::View