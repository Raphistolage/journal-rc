use::ndarray::{ArrayView};
use::mdspan_interop;
use ndarray::ShapeBuilder;

#[test] 
fn matrix_product_test() {
    let v = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
    let s = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
    let arr1 = ArrayView::from_shape((6).strides((1)), &v).unwrap();
    let arr2 = ArrayView::from_shape((6).strides((1)), &s).unwrap();

    println!("Orgininal ArrayViews : ");
    println!("Arr1 : {:?}", arr1);
    println!("Arr2 : {:?}", arr2);

    println!("Test matrix prod through shared struct : ");

    let _ = mdspan_interop::dot(arr1, arr2);

    // println!("After DeepCopy ArrayViews : ");
    // println!("Arr1: {:?}", arr1);
    // println!("Arr2 : {:?}", arr2);
}