use::ndarray::{ArrayView};
use::mdspan_interop2;
use ndarray::ShapeBuilder;

#[test] 
fn vector_product_test() {
    let v = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
    let s = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
    let arr1 = ArrayView::from_shape((6).strides(1), &v).unwrap();
    let arr2 = ArrayView::from_shape((6).strides(1), &s).unwrap();

    println!("Orgininal ArrayViews : ");
    println!("Arr1 : {:?}", arr1);
    println!("Arr2 : {:?}", arr2);

    println!("Test vector prod through shared struct : ");

    let result = mdspan_interop2::dot(&arr1, &arr2);
    println!("Result : {:?}", result);
    

    // mdspan_interop2::free_shared_array(result.as_ptr());

    // println!("Resulting scalar after freeing pointer : {:?}", result_array);
}