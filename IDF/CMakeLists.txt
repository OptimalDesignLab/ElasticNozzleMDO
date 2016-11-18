cmake_minimum_required(VERSION 3.6)
project(IDF)

# get Python and Boost library info from system
execute_process(COMMAND python-config --prefix
        OUTPUT_VARIABLE Python_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE)

execute_process(COMMAND python-config --ldflags
        OUTPUT_VARIABLE Python_LIBS
        OUTPUT_STRIP_TRAILING_WHITESPACE)

if (${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    set(LDFLAGS "-z,defs,-soname")
    set(CPP "-cpp")
elseif (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(LDFLAGS "-install_name")
    set(CPP "")
    if(NOT EXISTS "${Python_PREFIX}/include/python2.7/pyport.h.orig")
        execute_process(COMMAND cp ${Python_PREFIX}/include/python2.7/pyport.h ${Python_PREFIX}/include/python2.7/pyport.h.orig)
    endif()
    execute_process(COMMAND patch -p1 -N ${Python_PREFIX}/include/python2.7/pyport.h ../patch_pyport.h.diff)
endif()

set(CMAKE_CXX_FLAGS "${CPP} ${CMAKE_CXX_FLAGS} -g -fPIC -O3 -fpermissive")

set(Boost_LIBS "-lboost_python -lboost_program_options")

set(BASE_LIBS "-lm -lstdc++ -lblas -llapack")

set(CMAKE_SHARED_LINKER_FLAGS 
        "-shared -fPIC -Wl,${LDFLAGS},aerostruct_idf.so ${BASE_LIBS} ${Python_LIBS} ${Boost_LIBS}")

# source files to compile
set(SOURCE_FILES
        aerostruct_idf.cpp
        aerostruct_idf.hpp
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

# C preprocessor macros
add_definitions(-DNDEBUG)
add_definitions(-DBOOST_UBLAS_ENABLE_PROXY_SHORTCUTS)
add_definitions(-DBOOST_UBLAS_TYPE_CHECK=0)

# include dirs
set(CMAKE_INCLUDE_CURRENT_DIR ON)
include_directories(
        AFTER
        ${Python_PREFIX}/include
        ${Python_PREFIX}/include/python2.7
        ${Python_PREFIX}/lib/python2.7/site-packages/PyUblas-2013.1-py2.7-macosx-10.6-x86_64.egg/pyublas/include
        ${Python_PREFIX}/lib/python2.7/site-packages/numpy/core/include)

# library link dirs
link_directories(
        ${Python_PREFIX}/lib)

# build target
add_library(aerostruct_idf SHARED ${SOURCE_FILES})

set_target_properties(
        aerostruct_idf
        PROPERTIES
        PREFIX ""
        SUFFIX .so)