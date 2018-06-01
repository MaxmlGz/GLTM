#!/usr/bin/env bash

# We are verbose here on purpose

# First, compile all the cpp files
g++-6 -c -fPIC wrapper_python.cpp DataTypes.cpp GeneralFunctions.cpp

# Now, link them all into a shared library
g++-6 wrapper_python.o DataTypes.o GeneralFunctions.o -shared -Wl,-soname,libDataTypes.so -o libDataTypes.so -lgsl -lgslcblas -lm
# To make the compile issue work, I removed wrapper.o from the output.
# ...so wrapper_python.o

# Old compile
#g++-6 -c -fPIC wrapper_python.cpp -o wrapper_python.o
#g++-6 -shared -Wl,-soname,libDataTypes.so -o libDataTypes.so wrapper_python.o -lgsl -lgslcblas -lm


# --Header and cpp structures--

# GeneralFunctions.cpp
#     GeneralFunctions.h

# DataTypes.cpp
#     DataTypes.h
#     GeneralFunctions.h

# wrapper_python.cpp
#     wrapper_python.h
#     DataTypes.h
#     GeneralFunctions.h
