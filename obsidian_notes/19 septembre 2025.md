
# Fait
- Correction exo mod 4 Kokkos
- moitié module 5 Kokkos
- Utilisation d'un struct et opaque type pour passer/utiliser les types propres de Armadillo en bidirectionnel 
- Lecture UniquePtr, Box, SharedPtr avec Cxx
- Fini MarkDown pour justif Cxx, avec comparaison / wrapper C

# Observations
- Réduire considérablement le wrapper grace aux types opaques, mais wrapper **obligatoire**.
- Les types opaques servent à **transférer**, pas +.
- Cxx a pas l'air de consommer en perf (peut-être juste un peu, mais vérifier sur test grande échelle).

# TODO
- Utiliser une seule struct partagée (essayer)
- meilleur benchmark de Cxx


# Questions
- Kokkos::single, en particulier avec PerTeam

