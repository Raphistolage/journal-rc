use std::time::Instant;

use::ndarray::{ArrayView};
use::mdspan_interop::{matrix_product, from_shared, free_shared_array};

#[test] 
fn matrix_product_test() {
    let now = Instant::now();
    let v = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
    let s = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
    let arr1 = ArrayView::from_shape((2,2), &v).unwrap();
    let arr2 = ArrayView::from_shape((2,2), &s).unwrap();

    println!("Orgininal ArrayViews : ");
    println!("Arr1 : {:?}", arr1);
    println!("Arr2 : {:?}", arr2);

    println!("Test Matrix Vector prod through shared struct : ");

    let result = matrix_product(&arr1, &arr2);

    let result_array = from_shared(result);
    let elapsed = now.elapsed();
    println!("Resulting vector : {:?}", result_array);
    
    println!("Time elapsed : {}", elapsed.as_secs_f64());
    
    free_shared_array(result_array.as_ptr());

    println!("Resulting vector after freeing pointer : {:?}", result_array);
}