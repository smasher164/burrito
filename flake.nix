{
  description = "a build system and package manager for humans";
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    let supportedSystems = [
      "aarch64-linux"
      "i686-linux"
      "x86_64-linux"
    ]; in
    flake-utils.lib.eachSystem supportedSystems (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [];
        };
      in
      {
        devShell = pkgs.mkShell {
          buildInputs = [
            pkgs.rustc
            pkgs.cargo
            pkgs.clippy
          ];
        };
      });
}
