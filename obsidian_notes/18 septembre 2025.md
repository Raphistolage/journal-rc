## Fait
- Terminé la lecture du **module 4 de Kokkos**.
- Premier test avec **Cxx** pour interop Rust ↔ C++ :  
  - Pas évident au début mais ca a fini par fonctionner.
  - Mais **nécessite un wrapper**.
- Test avec **Armadillo** pour un produit de matrices :  
  - Le test fonctionne. Mais représentation des matrices en **column-major** -> rend le passage des données entre Rust et C++ complexe.
  -  Démarrage depos Git et Obsidian.

## Observations
- Wrappers obligatoires.  
- Standardiser les représentations des données entre les deux langages (en particulier avec les layouts de Kokkos).

## Demain
- Voir la **correction des exercices du module 4**.  
- Lecture du **module 5 de Kokkos**.  
- Tester de mettre en place des structs portables pour Cxx, et voir ce qui peut être porter.
- Finir MarkDown pour avantages/inconvénients de Cxx (comparer perfs, comparer tailles codes sans/avec (si on faisait avec wrappers à la place etc)).

## Questions
- Wrappers scalables ?  
- 
