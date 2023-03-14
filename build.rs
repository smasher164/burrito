extern crate bindgen;

use std::env;
use std::path::PathBuf;

fn main() {
    let bindings = bindgen::Builder::default()
        .header_contents("wrapper.h", r#"
            #include "ca_ext.h"
            #include "ca_field.h"
            #include "ca.h"
            #include "calcium.h"
            #include "ca_mat.h"
            #include "ca_poly.h"
            #include "ca_vec.h"
            #include "fexpr_builtin.h"
            #include "fexpr.h"
            #include "fmpz_mpoly_q.h"
            #include "qqbar.h"
            #include "utils_flint.h"
        "#)
        .blocklist_item("FP_NAN")
        .blocklist_item("FP_INFINITE")
        .blocklist_item("FP_ZERO")
        .blocklist_item("FP_SUBNORMAL")
        .blocklist_item("FP_NORMAL")
        .raw_line(r#"extern "C" { pub fn fmpq_init(x: *mut fmpq); }"#)
        .raw_line(r#"extern "C" { pub fn fmpq_clear(x: *mut fmpq); }"#)
        .raw_line(r#"extern "C" { pub fn fmpz_init(x: *mut fmpz); }"#)
        .raw_line(r#"extern "C" { pub fn fmpz_clear(x: *mut fmpz); }"#)
        .generate()
        .expect("unable to generate bindings");
        
        // Write the bindings to the $OUT_DIR/bindings.rs file.
        let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
        bindings
            .write_to_file(out_path.join("bindings.rs"))
            .expect("Couldn't write bindings!");
}