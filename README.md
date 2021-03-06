## ElasticNozzleMDO -- An Aero-Structural Analysis and Optimization Problem

A multi-disciplinary analysis and optimization code for a 2-D elastic nozzle problem.

The flow is analyzed using quasi-1D Euler equations, and the nozzle structure is modeled 
using Euler-Bernoulli beam elements.

The analysis code is written in C++ and an API is exposed to Python via Boost.

The multi-disciplinary problem is formulated using both the Multidisciplinary Feasible 
(MDF) and Individual Discipline Feasible (IDF) coupling architectures.

The optimization problem is solved using the Kona library.

### Dependencies

+ Kona Optimization Library:
https://github.com/OptimalDesignLab/Kona

+ Boost C++ (1.55 or newer):
https://www.boost.org/

+ Boost Numeric Library Bindings:
https://mathema.tician.de/software/boost-numeric-bindings/

+ PyUblas:
https://mathema.tician.de/software/pyublas/

### Installation

Clone the repository including submodules using:

```
$ git clone --recursive git@github.com:OptimalDesignLab/ElasticNozzleMDO.git
```

Then use the configuration script in either the IDF or the MDF folder before 
building the respective modules:

```
$ cd ElasticNozzleMDO/IDF
$ ./configure.py --boost-prefix=/usr/local
$ make -j4
```
For more options, see `./configure.py --help`.
