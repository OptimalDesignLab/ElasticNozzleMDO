# makefile for the Aero-Struct MDF Optimization project

SHELL = /bin/sh

.SUFFIXES:
.SUFFIXES: .cpp .o
.PHONY: default all tags pylib clean

# get python options
PYTHON=python
PYTHON-CONFIG=python-config
PYTHON_INCLUDES=$(shell $(PYTHON-CONFIG) --includes)
PYTHON_LIBS=-L$(shell $(PYTHON-CONFIG) --prefix)/lib $(shell $(PYTHON-CONFIG) --libs)

# get boost options
BOOST_PREFIX=/usr/local
BOOST_INCLUDES=-I$(BOOST_PREFIX)/include
BOOST_LIBS=-L$(BOOST_PREFIX)/lib -lboost_program_options -lboost_python

# compiler
CXX= g++

# c preprocessor options
#CPPFLAGS= -cpp -DDEBUG -DBOOST_UBLAS_ENABLE_PROXY_SHORTCUTS
CPPFLAGS= -cpp -DNDEBUG -DBOOST_UBLAS_ENABLE_PROXY_SHORTCUTS -DBOOST_UBLAS_TYPE_CHECK=0

# compiler options that may vary (user can change)
CXXFLAGS= -g -fPIC -fpermissive

# linker options
#LDFLAGS= -lm -lstdc++ -llapack -L. -L$(BOOST_ROOT)/lib -lboost_program_options  -L$#(KONA_ROOT) -lkona
LDFLAGS= $(PYTHON_LIBS) $(BOOST_LIBS) -lm -lstdc++ -llapack -lblas

# options that DO NOT vary
#ALL_CXXFLAGS= -I. $(CXXFLAGS) -I$(BOOST_ROOT)
ALL_CXXFLAGS= $(CXXFLAGS) -I. $(PYTHON_INCLUDES) $(BOOST_INCLUDES)

# directories
CFD_DIR=../Quasi1DEuler
CSM_DIR=../LECSM

# headers
CFD_HEADERS= $(CFD_DIR)/*.hpp
CSM_HEADERS= $(CSM_DIR)/*.hpp

# source files
CFD_SRC= $(CFD_DIR)/*.cpp
CSM_SRC= $(CSM_DIR)/*.cpp

# source and object file names
HEADERS= $(wildcard $(CFD_HEADERS) $(CSM_HEADERS) ../*.hpp)
SOURCES= $(wildcard $(CFD_SRC) $(CSM_SRC) ../*.cpp)
OBJS= $(SOURCES:.cpp=.o)

# implicit rule
%.o : %.cpp $(HEADERS) Makefile
	@echo "Compiling \""$@"\" from \""$<"\""
	@$(CXX) $(CPPFLAGS) $(ALL_CXXFLAGS) -o $@ -c $<

default: all
	@echo "Python includes:"
	@echo $(PYTHON_INCLUDES)
	@echo "Python libs:"
	@echo $(PYTHON_LIBS)
	@echo "Boost includes:"
	@echo $(BOOST_INCLUDES)
	@echo "Boost libs:"
	@echo $(BOOST_LIBS)

all: pylib

tags: $(HEADERS) $(SOURCES)
	@echo "Creating TAGS file for emacs"
	@find -maxdepth 2 -iname '*.hpp' -print0 -o \
	-iname '*.cpp' -print0 | xargs -0 etags

pylib: $(OBJS) aerostruct_mdf.o Makefile
	@echo "Compiling Python module for MDF"
	@$(CXX) -shared -fPIC -Wl,-z,defs,-soname,aerostruct_mdf.so \
	aerostruct_mdf.o $(OBJS) $(LDFLAGS) -o aerostruct_mdf.so

clean:
	@echo "deleting temporary, object, library and binary files"
	@rm -f *~
	@rm -f $(BINARIES)
	@rm -f $(OBJS) *.o
	@rm -f *.so*
