#!/usr/bin/env bash

g++ -c -fPIC np2cppyy.cpp -o np2cppyy.o
g++ -shared -Wl,-soname,lib.so -o lib.so  np2cppyy.o -lgsl #-lgslcblas -lm
