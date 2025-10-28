use std::path::Path;
use cc::Build;

mod file;
mod fs;

#[derive(Default)]
pub struct GeneratedCode {
    /// The bytes of a C++ header file.
    pub header: Vec<u8>,
    /// The bytes of a C++ implementation file (e.g. .cc, cpp etc.)
    pub implementation: Vec<u8>,
}

fn best_effort_remove(path: &Path) {
    use std::fs;

    if cfg!(windows) {
        // On Windows, the correct choice of remove_file vs remove_dir needs to
        // be used according to what the symlink *points to*. Trying to use
        // remove_file to remove a symlink which points to a directory fails
        // with "Access is denied".
        if let Ok(metadata) = fs::metadata(path) {
            if metadata.is_dir() {
                let _ = fs::remove_dir_all(path);
            } else {
                let _ = fs::remove_file(path);
            }
        } else if fs::symlink_metadata(path).is_ok() {
            // The symlink might exist but be dangling, in which case there is
            // no standard way to determine what "kind" of symlink it is. Try
            // deleting both ways.
            if fs::remove_dir_all(path).is_err() {
                let _ = fs::remove_file(path);
            }
        }
    } else {
        // On non-Windows, we check metadata not following symlinks. All
        // symlinks are removed using remove_file.
        if let Ok(metadata) = fs::symlink_metadata(path) {
            if metadata.is_dir() {
                let _ = fs::remove_dir_all(path);
            } else {
                let _ = fs::remove_file(path);
            }
        }
    }
}

pub fn write(path: impl AsRef<Path>, content: &[u8]){
    let path = path.as_ref();

    let mut create_dir_error = None;
    if fs::exists(path) {
        if let Ok(existing) = fs::read(path) {
            if existing == content {
                // Avoid bumping modified time with unchanged contents.
                return;
            }
        }
        best_effort_remove(path);
    } else {
        let parent = path.parent().unwrap();
        create_dir_error = fs::create_dir_all(parent).err();
    }

    fs::write(path, content);
}

pub fn build(rust_source_file: impl AsRef<Path>, cpp_source_file: impl AsRef<Path>, cpp_path: &Path) -> Build {
    let mut build = Build::new();
    build.cpp(true);
    build.include(cpp_source_file);

    let generated = generate_bridge(rust_source_file.as_ref()).unwrap();
    write(cpp_path, &generated.implementation);

    build
}

fn generate_bridge(rust_path: &Path) -> Result<GeneratedCode, syn::Error> {
    let source = match read_to_string(rust_path) {
        Ok(source) => source,
        Err(err) => panic!("read_to_string error"),
    };
    let source = source.as_str();
    let syntax: file::File = syn::parse_str(source)?;
    Ok(generate(syntax))
}

fn read_to_string(path: &Path) -> Result<String, ()> {
    let bytes = if path == Path::new("-") {
        fs::read_stdin()
    } else {
        fs::read(path)
    }.unwrap();
    match String::from_utf8(bytes) {
        Ok(string) => Ok(string),
        Err(_) => Err(()),
    }
}


pub fn generate(syntax: file::File) -> GeneratedCode {
    //TODO : GENERER LE CODE C++ EN BYTES
}