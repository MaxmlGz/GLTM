#!/usr/bin/env bash

g++-6 -c -fPIC foo.cpp -o foo.o
g++-6 -shared -Wl,-soname,libfoo.so -o libfoo.so  foo.o -lgsl -lgslcblas -lm
