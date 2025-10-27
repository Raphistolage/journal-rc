use::ndarray::{ArrayView};
use::mdspan_interop2::{matrix_product};

#[test] 
fn matrix_product_test() {
    let v: [u32; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
    let s: [u32; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
    let arr1 = ArrayView::from_shape((2,2), &v).unwrap();
    let arr2 = ArrayView::from_shape((2,2), &s).unwrap();

    println!("Orgininal ArrayViews : ");
    println!("Arr1 : {:?}", arr1);
    println!("Arr2 : {:?}", arr2);

    println!("Test Matrix Vector prod through shared struct : ");

    let result = matrix_product(&arr1, &arr2);

    println!("Result : {:?}", result);


    mdspan_interop2::free_shared_array(result.as_ptr());

    println!("Resulting scalar after freeing pointer : {:?}", result);
}