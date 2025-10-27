#pragma once
#include <cstddef>

extern "C" {

    void test_castor(void* my_ndarray);
    void show_struct_repr(void* my_ndarray, int length);
    void show_mdspan_repr(int length);

}