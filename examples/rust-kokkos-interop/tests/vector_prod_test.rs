use::ndarray::{ArrayView};
use::rust_kokkos_interop;
use ndarray::ShapeBuilder;

#[test] 
fn vector_product_test() {
    let v = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
    let s = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
    let arr1 = ArrayView::from_shape((6).strides(1), &v).unwrap();
    let arr2 = ArrayView::from_shape((6).strides(1), &s).unwrap();

    println!("Orgininal ArrayViews : ");
    println!("Arr1 : {:?}", arr1);
    println!("Arr2 : {:?}", arr2);

    println!("Test vector prod through shared struct : ");

    let shared_arr1 = rust_kokkos_interop::to_shared(&arr1);
    let shared_arr2 = rust_kokkos_interop::to_shared(&arr2);
    
    let result = rust_kokkos_interop::dot(shared_arr1, shared_arr2);


    // let result_array = rust_kokkos_interop::from_shared(result);
    println!("Resulting scalar : {:?}", result);
}