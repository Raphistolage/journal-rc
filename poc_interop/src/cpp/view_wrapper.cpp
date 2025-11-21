#include "view_wrapper.hpp"

extern "C" {
    SharedArrayView view_to_shared_c(const opaque_view::OpaqueView* opaqueView) {
        return opaqueView->view->view_to_shared();  
    }

    SharedArrayViewMut view_to_shared_mut_c(const opaque_view::OpaqueView* opaqueView) {
        return opaqueView->view->view_to_shared_mut();  
    }
}

