#! /usr/bin/env python

def var_to_option(caps_name):
    return caps_name.lower().replace("_", "-")

def var_kind(caps_name):
    last_underscore = caps_name.rfind("_")
    return caps_name[last_underscore+1:]

if __name__ == "__main__":
    import os
    import re
    import subprocess
    from sys import platform
    from optparse import OptionParser
    from distutils.sysconfig import get_python_inc

    usage = "usage: ./configure.py [options] arg1 arg2"
    parser = OptionParser(usage=usage)
    parser.add_option(
        "--with-cxx", 
        dest="cxx",
        default="g++",
        help="Path for C++ compiler.")
    parser.add_option(
        "--cxx-flags", 
        dest="cxx_flags",
        default="",
        help="CXX compiler flags.")
    parser.add_option(
        "--python-config", 
        dest="python_config",
        default="python-config",
        help="Path for python-config binary.")
    parser.add_option(
        "--boost-prefix", 
        dest="boost_prefix",
        default="/usr/local",
        help="Installation prefix for Boost C++.")

    options, args = parser.parse_args()

    try:
        import numpy
        numpy_incl = os.path.dirname(numpy.__file__) + '/core/include'
    except ImportError:
        raise ImportError("Cannot import dependency: Numpy")

    try:
        import pyublas
        pyublas_incl = os.path.dirname(pyublas.__file__) + '/include'
    except ImportError:
        raise ImportError("Cannot import dependency: PyUblas")

    if platform == "linux" or platform == "linux2":
        cpp = "-cpp"
    elif platform == "darwin":
        print "macOS detected! pyport.h needs a patch for Boost compatibility"
        choice = raw_input("continue? (y/n) ")
        if choice == "n":
            raise RuntimeError("Configuration terminated. Cannot install without patch.")
        elif choice == "y":
            pass
        else:
            raise ValueError("Invalid selection!")
        cpp = ""
        python_inc = get_python_inc()
        curr_path = os.path.dirname(os.path.abspath(__file__))
        os.chdir(python_inc)
        if not os.path.isfile("pyport.h.orig"):
            print "Backing up pyport.h as pyport.h.orig..."
            subprocess.call("cp pyport.h pyport.h.orig", shell=True)
        cmd = "patch -p1 -N < " + curr_path + "/../patch_pyport.h.diff"
        subprocess.call(cmd, shell=True)
        os.chdir(curr_path)

    substitutions = {
        "CXX" : options.cxx,
        "CPP" : cpp,
        "USER_CFLAGS" : options.cxx_flags,
        "PYTHON_CONFIG" : options.python_config,
        "BOOST_PREFIX" : options.boost_prefix,
        "NUMPY_INCL" : numpy_incl,
        "PYUBLAS_INCL" : pyublas_incl
    }

    print "Writing Makefile..."

    var_re = re.compile(r"\$\{([A-Za-z_0-9]+)\}")
    string_var_re = re.compile(r"\$str\{([A-Za-z_0-9]+)\}")
    for fname in ["Makefile"]:
        lines = open(fname+".in", "r").readlines()
        new_lines = []
        for l in lines:
            made_change = True
            while made_change:
                made_change = False
                match = var_re.search(l)
                if match:
                    varname = match.group(1)
                    l = l[:match.start()] + str(substitutions[varname]) + l[match.end():]
                    made_change = True

                match = string_var_re.search(l)
                if match:
                    varname = match.group(1)
                    subst = substitutions[varname]
                    if subst is None:
                        subst = ""
                    else:
                        subst = '"%s"' % subst

                    l = l[:match.start()] + subst  + l[match.end():]
                    made_change = True
            new_lines.append(l)

        file(fname, "w").write("".join(new_lines))

    print "DONE!"