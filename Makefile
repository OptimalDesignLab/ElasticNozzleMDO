# makefile for Kona library

SHELL = /bin/sh

.SUFFIXES:
.SUFFIXES: .cpp .o
.PHONY: default call tags MDA_test clean

# compiler
CXX= gcc 

# c preprocessor options
CPPFLAGS= -cpp -DDEBUG

# compiler options that may vary (user can change)
CXXFLAGS= -g 
BOOST_ROOT= /usr/include/boost
KONA_ROOT= /home/denera/Documents/Research/kona

# linker options
LDFLAGS= -lstdc++ -llapack -L. -lboost_program_options -lm -L$(KONA_ROOT) -lkona

# options that DO NOT vary
ALL_CXXFLAGS= -I. $(CXXFLAGS) -I$(BOOST_ROOT) -I$(KONA_ROOT)/src

# directories
CFD_DIR=./quasi_1d_euler
CSM_DIR=./linear_elastic_csm

# headers
CFD_HEADERS= $(CFD_DIR)/*.hpp
CSM_HEADERS= $(CSM_DIR)/*.hpp

# source files
CFD_SRC= $(CFD_DIR)/*.cpp
CSM_SRC= $(CSM_DIR)/*.cpp

# source and object file names
HEADERS= $(wildcard $(CFD_HEADERS) $(CSM_HEADERS))
HEADERS_ALL= $(HEADERS)
SOURCES= $(wildcard $(CFD_SRC) $(CSM_SRC))
SOURCES_ALL= $(SOURCES)
OBJS= $(SOURCES:.cpp=.o)
OBJS_ALL= $(SOURCES_ALL:.cpp=.o)
BINARIES= MDA_test.bin

# implicit rule
%.o : %.cpp $(HEADERS_ALL) Makefile
	@echo "Compiling \""$@"\" from \""$<"\""
	@$(CXX) $(CPPFLAGS) $(ALL_CXXFLAGS) -o $@ -c $<

default: all

all: $(BINARIES)

tags: $(HEADERS) $(SOURCES)
	@echo "Creating TAGS file for emacs"
	@find -maxdepth 2 -iname '*.hpp' -print0 -o \
	-iname '*.cpp' -print0 | xargs -0 etags

MDA_test.bin: $(OBJS) Makefile
	@echo "Compiling \""$@"\" from \""$(OBJS)"\""
	@$(CXX) -o $@ $(OBJS) $(LDFLAGS) 

clean:
	@echo "deleting temporary, object, and binary files"
	@rm -f *~
	@rm -f $(BINARIES)
	@rm -f $(OBJS) *.o
