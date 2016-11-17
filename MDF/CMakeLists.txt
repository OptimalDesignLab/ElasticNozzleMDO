cmake_minimum_required(VERSION 3.6)
project(MDF)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fPIC -O3 -fpermissive")

set(CMAKE_SHARED_LINKER_FLAGS "-shared -fPIC -Wl,-install_name,aerostruct_mdf.so -framework CoreFoundation -lpython2.7 -ldl -lm -lstdc++ -llapack -lblas -lboost_program_options -lboost_python")

set(SOURCE_FILES
        aerostruct_mdf.cpp
        aerostruct_mdf.hpp
        ../LECSM/test_problem/Makefile
        ../LECSM/test_problem/test_problem.cpp
        ../LECSM/1D_mesh_tools.cpp
        ../LECSM/1D_mesh_tools.hpp
        ../LECSM/lecsm.cpp
        ../LECSM/lecsm.hpp
        ../LECSM/matrix_tools.cpp
        ../LECSM/matrix_tools.hpp
        ../LECSM/output_tools.cpp
        ../LECSM/output_tools.hpp
        ../Quasi1DEuler/bspline.cpp
        ../Quasi1DEuler/bspline.hpp
        ../Quasi1DEuler/exact_solution.hpp
        ../Quasi1DEuler/hyperdual.cpp
        ../Quasi1DEuler/hyperdual.hpp
        ../Quasi1DEuler/inner_prod_vector.hpp
        ../Quasi1DEuler/nozzle.cpp
        ../Quasi1DEuler/nozzle.hpp
        ../Quasi1DEuler/quasi_1d_euler.cpp
        ../Quasi1DEuler/quasi_1d_euler.hpp
        ../Quasi1DEuler/sum_by_parts.cpp
        ../Quasi1DEuler/sum_by_parts.hpp
        ../aerostruct.cpp
        ../aerostruct.hpp
        ../constants.hpp
        ../krylov.cpp
        ../krylov.hpp
        ../krylov_def.hpp)

add_definitions(-DNDEBUG)
add_definitions(-DBOOST_UBLAS_ENABLE_PROXY_SHORTCUTS)
add_definitions(-DBOOST_UBLAS_TYPE_CHECK=0)

include_directories(
        AFTER
        /Users/denera/Developer/miniconda2/include
        /Users/denera/Developer/miniconda2/include/python2.7
        /Users/denera/Developer/miniconda2/lib/python2.7/site-packages/PyUblas-2013.1-py2.7-macosx-10.6-x86_64.egg/pyublas/include
        /Users/denera/Developer/miniconda2/lib/python2.7/site-packages/numpy/core/include)

link_directories(
        /Users/denera/Developer/miniconda2/lib)

add_library(aerostruct_mdf SHARED ${SOURCE_FILES})

set_target_properties(
        aerostruct_mdf
        PROPERTIES
        PREFIX ""
        SUFFIX .so)