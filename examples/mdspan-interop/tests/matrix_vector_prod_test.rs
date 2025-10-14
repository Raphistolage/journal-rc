use::ndarray::{ArrayView};
use::mdspan_interop;
use ndarray::ShapeBuilder;

#[test] 
fn matrix_vector_prod_test() {
    let v = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
    let s = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
    let arr1 = ArrayView::from_shape((2, 6), &v).unwrap();
    let arr2 = ArrayView::from_shape((6), &s).unwrap();

    println!("Orgininal ArrayViews : ");
    println!("Arr1 : {:?}", arr1);
    println!("Arr2 : {:?}", arr2);

    println!("Test Matrix Vector prod through shared struct : ");

    let result = mdspan_interop::matrix_vector_product(arr1, arr2);


    let result_array = mdspan_interop::from_shared(result);
    println!("Resulting vector : {:?}", result_array);
    
    mdspan_interop::free_shared_array(result_array.as_ptr());

    println!("Resulting vector after freeing pointer : {:?}", result_array);

}