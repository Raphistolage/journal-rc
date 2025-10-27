# Fait
- Casting entre ndarray::Array et std::mdspan => pas possible car owned / non owned
- Casting entre ndarray::ArrayView et std::mdspan => fonctionne
- Passer directement ndarray::ArrayView, interprété en mdspan sans soucis à par FFI not safe
- Wrapper ArrayView dans une struct avec repr(C), change pas le not safe.
- Tout passer en raw pointer => interprétation correcte dans certains cas

# Observations
- ArrayView <=> std::mdspan
- Pas FFI safe car pas repr(C)
- Pas besoin de casting

# TODO
- Rendre le passage  de ArrayView FFI safe.
# Questions
- 


