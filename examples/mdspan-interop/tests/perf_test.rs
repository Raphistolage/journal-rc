use std::time::Instant;

use ndarray::{Array, ArrayViewMut};
use::ndarray::{ArrayView};
use::mdspan_interop::{matrix_product, from_shared, free_shared_array};

#[test] 
fn perf_test_1() {

    {
        let mut v = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
        let mut s = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
        // let arr1 = ArrayView::from_shape((2,2), &v).unwrap();
        // let arr2 = ArrayView::from_shape((2,2), &s).unwrap();

        // println!("Orgininal ArrayViews : ");
        // println!("Arr1 : {:?}", arr1);
        // println!("Arr2 : {:?}", arr2);

        println!("Test Matrix Vector prod through shared struct : ");

        let mut tot_time = 0.0;
        let n = 1_000_000;

        for _ in 0..n {
            v[0] += 1.0;
            s[1] += 2.0; 
            let arr1 = ArrayView::from_shape((2,2), &v).unwrap();
            let arr2 = ArrayView::from_shape((2,2), &s).unwrap();

            let now = Instant::now();
            let result = matrix_product(&arr1, &arr2);
            tot_time += now.elapsed().as_secs_f64();

            let result_array = from_shared(result);


                
            free_shared_array(result_array.as_ptr());
        }
        
        println!("Total time elapsed in seconds: {}", tot_time);
        println!("Time elapsed in average per 2x2 matrix product in ns: {}", tot_time*1000.0);
    }


    println!("Test with only Rust : ");

    {
        let mut v = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
        let mut s = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];

        let mut tot_time = 0.0;
        let n = 1_000_000;

        for _ in 0..n {
            v[0] += 1.0;
            s[1] += 2.0; 
            let arr1 = ArrayView::from_shape((2,2), &v).unwrap();
            let arr2 = ArrayView::from_shape((2,2), &s).unwrap();

            let now = Instant::now();

            let mut result: ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 2]>> = Array::zeros((2,2));
            for i in 0..arr1.dim().0  {
                for j in 0..arr2.dim().1  {
                    let mut r: f64 = 0.0;
                    for k in 0..arr1.dim().1  {
                        r += arr1[(i, k)] * arr2[(k, j)];
                    }
                    result[[i,j]] = r;
                }
                tot_time += now.elapsed().as_secs_f64();
            }
        }
        
        println!("Total time elapsed in seconds: {}", tot_time);
        println!("Time elapsed in average per 2x2 matrix product in ns: {}", tot_time*1000.0);
    }

}