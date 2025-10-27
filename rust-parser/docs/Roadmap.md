
The goal is to create a parser lib that allows to write a function in rust, then parse it into a .cpp file (correct C++) that can be then compiled and used on runtime.

- Step 1 : Make a beacon #[cpp] that allows for writing a function that can then be parsed.
- Step 1.5 : prepare translation of symbols from rust to proper cpp.
- Step 2: recover the beaconed functions and re-write them in a separate file .cpp.
- Step 3: make linking (probably consider adding a header for the cpp file).
- Step 4: check compilation



### Sources

- La crate **syn** qui permet de lire. *
- **Quote** permet d'écrire <-- C'est la qu'il faut agir pour écrire non pas en Rust mais en C++, dans un autre fichier.
- **Cxx** qui permettrai de faire lien entre le code parsé et le rust.
- 