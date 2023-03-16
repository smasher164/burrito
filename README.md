Burrito
---

A build system and package manager for humans.

- HM + effects.
    - No value restriction needed.
    - Can possibly throw on == 0.
    - Tunneled effects ala https://dl.acm.org/doi/10.1145/3290318.
- Package type.
    - Reproducible builds.
    - Content-addressed?
    - Build steps are not shell-script wrappers but actually just using the stdlib's command execution facilities.
- A cross between Nix and Bazel: Constructive traces + Suspending
    - https://www.microsoft.com/en-us/research/uploads/prod/2020/04/build-systems-jfp.pdf