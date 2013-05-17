# makefile for linear_elastic_csm

SHELL = /bin/sh

.SUFFIXES:
.SUFFIXES: .cpp .o
.PHONY: default all tags test_prob clean

# compiler
CXX= gcc 

# c preprocessor options
CPPFLAGS= -cpp

# compiler options that may vary (user can change)
CXXFLAGS= 
#BOOST_ROOT= /usr/local/boost_1_47_0

# linker options
# NOTE: -L is for linking, -Wl,-rpath is for loading
#LDFLAGS= -lstdc++ -L$(BOOST_ROOT)/stage/lib -lboost_program_options
LDFLAGS= -lstdc++

# options that DO NOT vary
#ALL_CXXFLAGS= -I. $(CXXFLAGS) -I $(BOOST_ROOT)
ALL_CXXFLAGS= -I. $(CXXFLAGS)

# source, object, and executable file names
HEADERS= $(wildcard *.hpp)
SOURCES= $(wildcard *.cpp)
OBJS= $(SOURCES:.cpp=.o)
BINARIES= test_prob.bin

# implicit rule
%.o : %.cpp $(HEADERS) Makefile
	@echo "Compiling \""$@"\" from \""$<"\""
	@$(CXX) $(CPPFLAGS) $(ALL_CXXFLAGS) -o $@ -c $<

default: all

all: $(BINARIES)

tags: $(HEADERS) $(SOURCES)
	@echo "Creating TAGS file for emacs"
	@find -maxdepth 2 -iname '*.hpp' -print0 -o \
	-iname '*.cpp' -print0 | xargs -0 etags

test_prob.bin: $(OBJS) Makefile
	@echo "Compiling \""$@"\" from \""$(OBJS)"\""
	@$(CXX) -o $@ $(OBJS) $(LDFLAGS) 

clean:
	@echo "deleting temporary, object, and binary files"
	@rm -f *~
	@rm -f $(BINARIES)
	@rm -f $(OBJS) *.o