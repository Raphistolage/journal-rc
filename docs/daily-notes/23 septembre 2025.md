# Fait
- Terminé utilisation shared struct avec Cxx
- Utilisation des types de Cxx (rust::Vec), conversion ownership (const & pour & etc.)
- Comparaison des temps pour appels de fonctions à travers le FFI
- Kokkos module 7

# Observations
- Passer des objets plutot que des refs => x4 en temps, donc mauvais.
- Difficulté sur conversion des types (&, &mut, \*mut etc)
- On passe de 0.00144s à 0.00132s pour éxécuter programme +- similaire, simplement en changeant d'un appel C++ -> C++ à C++ -> Rust.
- On va pouvoir utiliser SimpleKernelTimer pour voir si nos appels depuis Rust sont effiaces (si on passe + de temps dans les kernels que ailleurs).

# TODO
- Module 8
- Faire + de Cxx (tester d'autre types etc, creuser encore type opaques)


# Questions
- HPCToolkit ?


