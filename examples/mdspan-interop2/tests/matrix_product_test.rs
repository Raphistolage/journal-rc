use::ndarray::{ArrayView};
use::mdspan_interop2::{matrix_product};

#[test] 
fn matrix_product_test() {
    let v: [f32; 12] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
    let s: [f32; 12] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
    let arr1 = ArrayView::from_shape((2,2), &v).unwrap();
    let arr2 = ArrayView::from_shape((2,2), &s).unwrap();

    println!("Orgininal ArrayViews : ");
    println!("Arr1 : {:?}", arr1);
    println!("Arr2 : {:?}", arr2);

    println!("Test Matrix Vector prod through shared struct : ");

    let result = matrix_product(&arr1, &arr2);

    println!("Result : {:?}", result);
}