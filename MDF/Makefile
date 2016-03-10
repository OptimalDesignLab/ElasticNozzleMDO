# makefile for the Aero-Struct MDF Optimization project

SHELL = /bin/sh

.SUFFIXES:
.SUFFIXES: .cpp .o
.PHONY: default all tags pylib MDA_test MDF_optimize clean

# compiler
CXX= g++

# c preprocessor options
#CPPFLAGS= -cpp -DDEBUG -DBOOST_UBLAS_ENABLE_PROXY_SHORTCUTS
CPPFLAGS= -cpp -DNDEBUG -DBOOST_UBLAS_ENABLE_PROXY_SHORTCUTS -DBOOST_UBLAS_TYPE_CHECK=0

# compiler options that may vary (user can change)
CXXFLAGS= -g -fPIC -fpermissive

# linker options
#LDFLAGS= -lm -lstdc++ -llapack -L. -L$(BOOST_ROOT)/stage/lib -lboost_program_options  -L$#(KONA_ROOT) -lkona
LDFLAGS= -lboost_program_options -lboost_python -lm -lstdc++ -llapack -lblas -lpython2.7

# options that DO NOT vary
#ALL_CXXFLAGS= -I. $(CXXFLAGS) -I$(BOOST_ROOT) -I$(KONA_ROOT)/src
ALL_CXXFLAGS= -I. $(CXXFLAGS) -I/usr/include/python2.7

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
BINARIES= MDA_test.bin MDF_optimize.bin
LIBS= libaerostructmdf.so.1

# implicit rule
%.o : %.cpp $(HEADERS) Makefile
	@echo "Compiling \""$@"\" from \""$<"\""
	@$(CXX) $(CPPFLAGS) $(ALL_CXXFLAGS) -o $@ -c $<

default: all

all: $(BINARIES) pylib

tags: $(HEADERS) $(SOURCES)
	@echo "Creating TAGS file for emacs"
	@find -maxdepth 2 -iname '*.hpp' -print0 -o \
	-iname '*.cpp' -print0 | xargs -0 etags

pylib: $(OBJS) aerostructMDF.o Makefile
	@echo "Compiling Python module for MDF"
	@$(CXX) -shared -fPIC -Wl,-z,defs,-soname,aerostructmdf.so.1 \
	-o aerostructmdf.so.1.0 aerostructMDF.o $(OBJS) $(LDFLAGS)
	@ln -s -f aerostructmdf.so.1.0 aerostructmdf.so
	@ln -s -f aerostructmdf.so.1.0 aerostructmdf.so.1

MDA_test.bin: $(OBJS) MDA_test.o Makefile
	@echo "Compiling \""$@"\" from \""$(OBJS)"\""
	@$(CXX) -o $@ MDA_test.o $(OBJS) $(LDFLAGS)

MDF_optimize.bin: $(OBJS) MDF_optimize.o Makefile
	@echo "Compiling \""$@"\" from \""$(OBJS)"\""
	@$(CXX) -o $@ MDF_optimize.o $(OBJS) $(LDFLAGS)

clean:
	@echo "deleting temporary, object, library and binary files"
	@rm -f *~
	@rm -f $(BINARIES)
	@rm -f $(OBJS) *.o
	@rm -f *.so*
