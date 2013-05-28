# makefile for quasi_1d_euler

SHELL = /bin/sh

.SUFFIXES:
.SUFFIXES: .cpp .o
.PHONY: default clean all

# compiler
CXX= gcc 

# c preprocessor options
#CPPFLAGS= -cpp -DNDEBUG -DBOOST_UBLAS_ENABLE_PROXY_SHORTCUTS -DBOOST_UBLAS_TYPE_CHECK=0
CPPFLAGS= -cpp -DDEBUG -DBOOST_UBLAS_ENABLE_PROXY_SHORTCUTS

# compiler options that may vary (user can change)
#CXXFLAGS= -g
CXXFLAGS= 
BOOST_ROOT= $(BOOST_HOME)
KONA_ROOT= $(KONA_HOME)

# linker options
# NOTE: -L is for linking, -Wl,-rpath is for loading
#LDFLAGS= -lstdc++ -L$(KONA_ROOT) -Wl,-rpath,$(KONA_ROOT) -lkona
LDFLAGS= -lm -lstdc++ -L$(BOOST_ROOT)/stage/lib -lboost_program_options \
	-L$(KONA_ROOT) -lkona

# options that DO NOT vary
ALL_CXXFLAGS= -I. $(CXXFLAGS) -I $(BOOST_ROOT) -I $(KONA_ROOT)/src

# directories
CFD_DIR=../quasi_1d_euler
CSM_DIR=../linear_elastic_csm

# headers
CFD_HEADERS= $(wildcard $(CFD_DIR)/*.hpp)
CSM_HEADERS= $(wildcard $(CSM_DIR)/*.hpp)
HEADERS_ALL= $(CFD_HEADERS) $(CSM_HEADERS)

# source files
CFD_SOURCES= $(wildcard $(CFD_DIR)/*.cpp)
CSM_SOURCES= $(wildcard $(CSM_DIR)/*.cpp)
SOURCES_ALL= $(CFD_SOURCES) $(CSM_SOURCES)

# object files
OBJS_ALL= $(SOURCES_ALL:.cpp=.o)

# executables
BINARIES= IDF_test.bin

# implicit rule
%.o : %.cpp $(HEADERS_ALL) Makefile
	@echo "Compiling \""$@"\" from \""$<"\""
	@$(CXX) $(CPPFLAGS) $(ALL_CXXFLAGS) -o $@ -c $<

default: all

all: $(BINARIES)

IDF_test.bin: $(OBJS_ALL) idf.o Makefile
	@echo "Compiling \""$@"\" from \""$(OBJS_ALL)"\""
	@$(CXX) -o $@ idf.o $(OBJS_ALL) $(LDFLAGS)

clean:
	@echo "deleting temporary, object, and binary files"
	@rm -f *~
	@rm -f $(BINARIES)
	@rm -f $(OBJS_ALL)