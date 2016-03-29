#Multi-disciplinary Elastic Nozzle Problem

This is an aero-structural analysis code for an elastic nozzle problem.

The structural analysis is done via 2D beam-elements. The flow analysis is
quasi-1D Euler.

The repository includes both MDF and IDF coupling.

The solver is intended to be used with the Kona optimization library.

##Dependencies

+ [Boost 1.55](http://www.boost.org/users/history/version_1_55_0.html)

    + Boost.Python
    + Boost.Program_options

+ [Python 2.7](https://www.python.org/download/releases/2.7/)

+ [Kona Optimization Library](https://github.com/OptimalDesignLab/Kona)

##Compiling

Edit the ``Makefile`` so that ``BOOST_PREFIX`` points to the Boost install
directory. Invoke ``Makefile`` to build the shared object.

``opt_run.py`` will run the Kona optimization or verification.
