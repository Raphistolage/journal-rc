use::ndarray::{ArrayViewMut, ArrayView};
use::mdspan_interop;
use ndarray::ShapeBuilder;

#[test] 
fn create_shared_test() {
    // let mut arr = Array2::<i32>::ones((2, 6));
    let mut v = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let s = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
    let mut arr1 = ArrayViewMut::from_shape((2, 6).strides((1,2)), &mut v).unwrap();
    let arr2 = ArrayView::from_shape((2, 6).strides((1, 2)), &s).unwrap();

    println!("Orgininal ArrayViews : ");
    println!("Arr1 : {:?}", arr1);
    println!("Arr2 : {:?}", arr2);

    println!("Test cast through shared struct : ");

    mdspan_interop::deep_copy(&mut arr1, &arr2);

    println!("After DeepCopy ArrayViews : ");
    println!("Arr1: {:?}", arr1);
    println!("Arr2 : {:?}", arr2);

    // println!("Strides : {:?}", arr.strides());

    // let my_instant_owned = to_shared_owned(arr);
    // println!("Shape from Shared : {:?}", my_instant_owned.shape);

    // let my_instant = to_shared(&arr2);
    // println!("Shapes from Shared : {:?}", my_instant.shape);
    // test_fn(&mut arr);
}