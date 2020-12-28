# XAMG
**XAMG** is a library for solving sparse systems of linear algebraic equations with multiple right-hand side vectors. The library focuses on multiple solutions of systems with constant matrices and contains highly optimized methods implementation for the solution phase only.

The library contains the implementation of:

- a set of BiCGStab methods, including the merged formulations [1];
- algebraic multigrid methods (the matrix hierarchy is built using **hypre** library);
- Jacobi and Gauss-Seidel-like methods;
- Chebyshev polynomials.

[1] B. Krasnopolsky, Revisiting Performance of BiCGStab Methods for Solving Systems with Multiple Right-Hand Sides, [arXiv:1907.12874](https://arxiv.org/abs/1907.12874), 2019

The following features are implemented in **XAMG** library:
- mixed precision floating point calculations;
- dynamic indices compression (for 1 or 2-byte integers);
- per-level multigrid parameters specification.

**{TODO:}** The following features are planned to be implemented in the code:
- native support for 64-bit integer indices;
- adaptive data representation using various matrix storage formats for each matrix block.

The code is implemented in C++ (based on C++11 standard), which allows on par with data alignment to provide vectorization for all the main computational blocks. The current version of the code supports hybrid three-level hierarchical parallel programming model (MPI+POSIX shared memory). Optional use of GPU accelerators (multiGPU mode) is planned to be implemented in the future.

The library provides API for usage with applications written in C. The Fortran API is planned to be developed in the future.

# Building and running an example on a new system

1) `git clone --recursive XAMG...`
> *NOTE: It is important to add option `--recursve` to automatically clone dependent repos*
2) Choose right compilers (manually tuning PATH or using 'module', it depends)
3) `cd ThirdParty` && `<set CC,CXX,MPICC,MPICXX variables to show right compilers>` && `./dnb.sh` && `cd ..`
4) `cd examples/cpp`
5) `make`
> *It will fail and say, that there is no specific config*

6) **option a):**\
   Create a new config by copying the `generic.inc` contents with a right name + optional edits. Best name for a new config is:\
`<username>-<hostname>.inc` because this file is included by make automatically, if it exists. \
  **option b):**\
   Run: `make CONFIG=generic`
7) Make a specific block in `compflags.inc` and tune `CXXOPTIMIZE_{XXX}` and other variables to get best fitting options for the machine/compiler/environment combination.
8) Create a specific `env.sh` file in a XAMG source root, which implements 2 bash functions: `env_init_global` and `env_init`. You may put some machine-specific settings as environment variables there. For example you can put there all `module` utility calls or PATH settings from step 2, also you can save there right settings for variables: `CC`, `CXX`, `MPICC`, `MPICXX`, `CUDA_HOME`, `INSTALL_DIR`, `MAKE_PARALLEL_LEVEL`, `DNB_NOCUDA`, `PACKAGE_VERSIONS`. The `env.sh` file is loaded by `dnb.sh` automatically, if exists. This results in full automation of specific tunings for prerequisites download and build procedures.

# Makefile options:
`make [<target> <target> ...] [variable=value variable=value ...]`

## Usage examples:
```
make clean cpp example BUILD=Release
make BUILD=Debug XAMG_USER_FLAGS="-DXAMG_NV=2"
make WITH_CUDA=TRUE
make CONFIG=generic CXX=icpc MPICXX=/opt/openmpi-1.5.0/bin/mpicxx
```
## Variables for make:
- `BUILD=Release|Debug`  (default is: `Debug`)
- `XAMG_USER_FLAGS="<any_compiler_flags>"`  (default is: `<empty>`)
- `WITH_SEPARATE_OBJECTS=TRUE|FALSE` (default is: `FALSE`)
- `WITH_LIMITED_TYPES_USAGE=TRUE|FALSE` (default is: `TRUE`)
- `WITH_CUDA=TRUE|FALSE` (default is: `FALSE`)
- `WITH_GCOV=TRUE|FALSE` (default is: `FALSE`) NOT IMPLEMENTED YET!
- `WITH_GPROF=TRUE|FALSE` (default is: `FALSE`)
- `WITH_CUDA_PROFILE=TRUE|FALSE` (default is: `FALSE`) NOT IMPLEMENTED YET!
- `WITH_ITAC=TRUE|FALSE` (default is: `FALSE`)
- `CONFIG=<config-file-name>.inc` (default is: `<username>-<hostname>.inc` placed either in current dir or `Make.inc/`)
- `MACHINEID=<machineid>`   (default is: `<empty>`. Is supposed to be set in config. Can be GENERIC to pick up generic options.
> *NOTE: the `machineid` string is just a suffix for some variables group in compflags.inc. One can introduce any new machineid.*
- `CXX=<compiler>` (default: supposed to be set in config.inc, otherwise is replaced with `c++`)
- `CC=<compiler>` (default: supposed to be set in config.inc, otherwise is replaced with `cc`)
- `MPICXX=<compiler>` (default: supposed to be set in config.inc, otherwise is replaced with `mpicxx`)
- `MPICC=<compiler>` (default: supposed to be set in config.inc, otherwise is replaced with `mpicc`)
- `NV_CXX=<compiler>` (default: supposed to be set in config.inc, otherwise is replaced with `nvcc`)

## Documentation

[Comprehensive build documentation]: https://gitlab.com/xamg/xamg/-/wikis/docs/XAMG_build_guideline

[List of numerical methods' parameters]: https://gitlab.com/xamg/xamg/-/wikis/docs/XAMG_params_reference


## Acknowledgements

The code development was supported by the Russian Science Foundation (RSF) Grant No. 18-71-10075.
