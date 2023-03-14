{
  description = "a build system and package manager for humans";
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    rust-overlay.url = "github:oxalica/rust-overlay";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, rust-overlay, flake-utils, ... }:
  # outputs = { self, nixpkgs, flake-utils, ... }:
    let supportedSystems = [
      "aarch64-linux"
      "i686-linux"
      "x86_64-linux"
    ]; in
    flake-utils.lib.eachSystem supportedSystems (system:
    let
        overlays = [ (import rust-overlay) ];
        # overlays = [ ];
        pkgs = import nixpkgs {
          inherit system overlays;
          # inherit system;
        };
      in
      with pkgs;
      {
        devShell = pkgs.mkShell {
          buildInputs = [
            (rust-bin.selectLatestNightlyWith (toolchain: toolchain.default))
            rustPlatform.bindgenHook
            pkgs.flint
            pkgs.arb
            pkgs.gmp
            pkgs.mpfr
            pkgs.mpir
            pkgs.antic
            pkgs.calcium
          ];
          nativeBuildInputs = [
            
          ];
          shellHook = ''
            export CALCIUM_PATH="${pkgs.calcium}"
            export FLINT_PATH="${pkgs.flint}"
            export BINDGEN_EXTRA_CLANG_ARGS="$BINDGEN_EXTRA_CLANG_ARGS I${pkgs.flint}/include/flint -I${pkgs.calcium}/include/calcium"
            export RUSTFLAGS="-lcalcium -larb -lantic -lflint -lmpfr -lgmp"
          '';
        };
      });
}
