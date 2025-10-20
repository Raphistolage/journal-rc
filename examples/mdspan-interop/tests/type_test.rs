use std::time::Instant;

use mdspan_interop::to_shared;
use::ndarray::{ArrayView};
use::mdspan_interop::{matrix_product, from_shared, free_shared_array};

#[test] 
fn type_test() {

    #[derive(Debug)]
    struct Person<'a>{
        name: &'a str,
        age: u8,
        lucky_numbers: &'a [u8],
        secret_ptr: *const f64,
    }

    let secretA = 42.0;
    let secretB = 13.0;   
    let personA = Person{name: "Raphael", age: 21, lucky_numbers: &[1,2,3,4], secret_ptr: &secretA};
    let personB = Person{name: "Samuel", age: 22, lucky_numbers: &[1,5,7,8], secret_ptr: &secretB};

    let v = [personA, personB];

    let arr = ArrayView::from_shape(2, &v).unwrap();

    println!("Orgininal ArrayViews : ");
    println!("Arr1 : {:?}", arr);
    
    let shared_arr = to_shared(arr);
}