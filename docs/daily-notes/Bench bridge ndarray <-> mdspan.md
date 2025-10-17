
The following images show the total elapsed time and the average time per product for 2x2 matrices products, iterated 1_000_000 times.

The goal here is to see and compare the time performances of doing a matrix product using std::mdspan in C++ versus when initializing, converting and calling C++ side all from the Rust side.

### C++ : 

![C++PerfO3](./images/C++PerfO3.png)  
*C++ performance with O3 optimisations*
Average time : 30ns

![C++PerfO0](./images/C++PerfO0.png)  
*C++ performance with O0 optimisations*
Average time : 582ns

### Rust : 

![RustPerfO3](./images/RustPerfO3.png)
*Rust performance with O3 optimisations for C++ side compilation*
Average time : 585ns

![RustPerfO0](./images/RustPerfO0.png)
*Rust performance with O0 optimisations for C++ side compilation*
Average time : 1720ns


### Observations

The point is pretty straight forward : **the rust bridge adds a lot of overhead** (x20 in O3 and ~x3 in O0).

We here can see two important things :
- The difference between the two average times of the Rust code indicates that **a significant part of the code on the C++ side of the bridge can be optimized** by the compiler.
- The difference between the two ratios of performance gain with '-O3' and '-O0' reveals that most of the **operations done with the Rust/C++ bridge are not optimizable by clang**, which explains the significant difference between the x3 and x20.

### Things to try

To tackle these performance losses, the next steps could be to : 

- **Move all casting operations to C++ side** to benefit from eventual optimization when setting O3 (however we already have the Rust -> C++ way conversions on the C++ side, so we can only put the C++ -> Rust way conversions on the C++ to make a difference now)
- **Multiply bigger matrices**,  see the impact on memory and time performance.
- Most importantly : **pass only reference or pointers**, to avoid copies. (However this has the downside of not respecting know implementation of Blas' matrix_product, who is supposed to take objects as parameters, not refs).
