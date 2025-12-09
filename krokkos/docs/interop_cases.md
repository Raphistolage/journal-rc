**All the cases below should, in the end, be usable with any data type (i32, f64, u8 ...). For now only f64/double**
**All the kernels should be able to run on a Device execution space (views and mdspans have to be able to live on device memory space)**

- [x] Instantiate Rust variable (that implements toShared) 
    - cast to SharedArray (non-owning)
    - call C++ function on it 
    - cast it into std::mdspan on Host (non-owning)
    - call Kokkos kernel on it (ExecSpace = HostSpace) 
    - read modified value from rust.

https://github.com/Raphistolage/journal-rc/blob/e8c6f2e592f050d2f19661df79695b1c965869fb/poc_interop/src/mdspan_interop/ops.rs#L110-L126
&& 
https://github.com/Raphistolage/journal-rc/blob/e8c6f2e592f050d2f19661df79695b1c965869fb/poc_interop/src/mdspan_interop/ops.rs#L43-L53

- [X] Instantiate Rust variable (that implements toSharedMut) 
    - cast to SharedArrayMut (non-owning, mutable) or ArrayView (owning, consuming the variable)
    - move the data to device 
    - call C++ function on it 
    - cast it into Kokkos::View on Device or std::mdspan on Host (own a copy on device)
    - call Kokkos kernel on it (ExecSpace = Cuda,HIP...) 
    - copy to SharedArrayMut (non-owning) or cast to ArrayView (owning)
    - read modified value from Rust OR receive new ArrayView on rust side.
 
https://github.com/Raphistolage/journal-rc/blob/3e84e38fc6feea6d7ac08686f2b157be7e071447/poc_interop/src/mdspan_interop/ops.rs#L144-L162

- [ ] Instantiate Rust variable (with trait ToShared) 
    - cast to SharedArray (non-owning)
    - call C++ function / kokkos kernel on it, by REFERENCE (non mutable data in the SharedArray), but tries to change data anyway (breaking rust's warranties) 
    - expect error or not possible.
    
https://github.com/Raphistolage/journal-rc/blob/3e84e38fc6feea6d7ac08686f2b157be7e071447/poc_interop/src/mdspan_interop/ops.rs#L164-L180

- [x] Instantiate Rust variable (with trait ToSharedMut) 
    - cast to SharedArrayMut (non-owning, mutable)
    - call C++ function / kokkos kernel on it, by REFERENCE (mutable data in the SharedArrayMut) 
    - read back my data in rust, expecting specific modifications. 
    https://github.com/Raphistolage/journal-rc/blob/613b73077afe7113d515038736fcc945b4117233/poc_interop/src/mdspan_interop/ops.rs#L139-L159

- [ ] From Rust instantiate Kokkos::View (call to constructor, DEVICE) with vec from rust side (data on heap).
    -  Copy vec data to device view. (device view owns its data, vec is consumed by the call to constructor)
    - Execute kernel on device exec_space
    - Recover data on rust side, pass it to rust function
    - send back to device kernel on C++ side

- [x] From Rust instantiate Kokkos::View (call to constructor) with vec from rust side (data on heap), vec is consumed.
    - If Kokkos::View on device, need to deep_copy data.
    - receive OpaqueView {
        view: UniquePtr<IView>,

        size: u32,

        rank: u32,

        shape: Vec<i32>,

        mem_space: MemSpace,

        layout: Layout,
} (IView.view, which is a Kokkos::view, owns the data sent)
    - call Kokkos kernel on it 
    - cast OpaqueView.view to kokkos::view & execute kernel 
    - receive result (OpaqueView or other).
    - Data lives as long as OpaqueView lives.

https://github.com/Raphistolage/journal-rc/blob/dcd62f3567393295505951d6fb10f4ed56738e71/poc_interop/src/rust_view/ops.rs#L56-L76
&&
https://github.com/Raphistolage/journal-rc/blob/dcd62f3567393295505951d6fb10f4ed56738e71/poc_interop/src/cpp/view_wrapper.cpp#L312-L341

- [ ] Write a C++ function using kernels and views : void func(view1, view2) {Kokkos::parallel_for(... {})} 
    - call it from fn rust_func(array1, array2) 
    - array1.toShared 
    - ffi::call_kokkos_func(shared_array1, shared_array2) 
    - view1 = shared_array1.from_shared() 
    - func(view1, view2)                (Link with rust macro ?).


```
C++ :
double func(SharedArray shared_view1, shared_view2) {
    auto view1 = from_shared(shared_view1);
    ...
}

Rust : 
\#[from_cpp]
fn func(arr1: &impl toShared, arr2: &impl toShared) -> f64;

Generate ==>  

extern "C"  {
    func(shared_array1: SharedArray, shared_array2:... ) -> f64;
}

fn func(arr1: &impl toShared, arr2: &impl toShared) -> f64 {
let shared_arr1 = arr1.to_shared();
... shared_arr2 = ...
return func(shared_arr1, shared_arr2);
}
```

- [x] Instantiate std::mdspan (non-owning) on C++ side with some data.
    - cast to SharedArray (toShared, non-owning) 
    - call Rust function on it 
    - cast it into ndarray (or anything that implements fromShared, non-owning) 
    - execute Rust function on it (read-only, immutable)
    - Receive result on C++ side (ArrayView or anything), original data wasn't modified.  

https://github.com/Raphistolage/journal-rc/blob/173ff2285faebba416d71ba1ce5a83c143943b48/poc_interop/src/cpp/mdspan_interop.cpp#L325-L336

- [x] Instantiate std::mdspan (non-owning) on C++ side with some data.
    - cast to SharedArrayMut (toSharedMut, non-owning, mutable) 
    - call Rust function on it 
    - cast it into ndarray (or anything that implements fromSharedMut, non-owning, mutable) 
    - execute Rust function on it (can mutate inner data)
    - Receive result on C++ side (ArrayView or anything), original data may have been modified.  

https://github.com/Raphistolage/journal-rc/blob/173ff2285faebba416d71ba1ce5a83c143943b48/poc_interop/src/cpp/mdspan_interop.cpp#L338-L351

- [ ] From C++ instantiate ndarray (non-owning) with data that lives on C++ side
    - receive OpaqueArray 
    - call Rust function on it (read-only, data is owned C++ side) 
    - cast opaqueArray & execute Rust function (read-only, data is owned C++ side) 
    - receive result, original data wasn't modified.



### Summary Table

| Data instantiate \ Owning | Rust | C++ |
|---------------------------|-------|-----|
| Rust | impl ToShared (Mut) <br> \| <br> SharedArray (Mut) (Non Owning) <br> \| <br> Mdspan (non-owning) / Kokkos::View (Device, Own copy) | fn create_opaque_view(data, metadata...) <br> \| <br> OpaqueView2DF64 (Owning) <br> \| <br> Kokkos::View (Owning) |
| C++  |  | mdspan / Kokkos::View <br> \| <br> to_shared<> <br> \| <br> SharedArray (Mut) (Non owning) <br> \| <br> impl FromShared (Non owning) |
