use std::time::Instant;
use::ndarray::{ArrayView};
use::mdspan_interop;
use ndarray::ShapeBuilder;

#[test] 
fn vector_product_test() {
    let now = Instant::now();
    let v = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
    let s = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
    let arr1 = ArrayView::from_shape((6).strides(1), &v).unwrap();
    let arr2 = ArrayView::from_shape((6).strides(1), &s).unwrap();

    println!("Orgininal ArrayViews : ");
    println!("Arr1 : {:?}", arr1);
    println!("Arr2 : {:?}", arr2);

    println!("Test vector prod through shared struct : ");

    let result = mdspan_interop::dot(arr1, arr2);


    let result_array = mdspan_interop::from_shared(result);
    let elapsed = now.elapsed();
    println!("Resulting scalar : {:?}", result_array);
    println!("Time elapsed : {}", elapsed.as_secs_f64());
    
    mdspan_interop::free_shared_array(result_array.as_ptr());

    println!("Resulting scalar after freeing pointer : {:?}", result_array);
}