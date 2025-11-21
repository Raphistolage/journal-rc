#include "functions.hpp"

namespace functions {
    
}

extern "C" {
    SharedArrayView view_to_shared_c(const OpaqueView* opaqueView) {
        return opaqueView->view->view_to_shared();  
    }

    SharedArrayViewMut view_to_shared_mut_c(const OpaqueView* opaqueView) {
        return opaqueView->view->view_to_shared_mut();  
    }
}

