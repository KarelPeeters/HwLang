#!/usr/bin/env bash
set -ue

mkdir -p build
clang++ -I../rust/ignored main.cpp -o build/main
./build/main
