# Fait
- Utilisation de la struct intermediaire
- Mise en place de deep copy en 1D
- Passage de tout en référence pour garder ownership coté Rust.
- Deep copy en 2D (stride wise)
- Changé la struct pour avoir des raw ptr pour data => plus de slice

# Observations
- Pas besoin de renvoyer des trucs, tout en ref, inplace
- passer des raw ptr peut casser la repr de la struct (bizarre)
- Si le stride imposé rend les  données non contigues, on peut pas to_slice, et donc tout casse.
- Passé en raw ptr, on peut utiliser, mais représentation pas bonne. => fixed en multipliant chaque index par stride[i].

# TODO
- Ajouter d'autre fonctions
# Questions
- 
