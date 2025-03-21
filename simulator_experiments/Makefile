PYTHON_INCLUDES = $(shell python3-config --includes)
EXTENSION_SUFFIX = $(shell python3-config --extension-suffix)

all: build/main	build/module$(EXTENSION_SUFFIX)

clean:
	rm -rf build

run_main: build/main
	./build/main

build/main: ../rust/ignored/simulator.cpp wrapper.cpp main.cpp
	mkdir -p build
	clang++ -O3 \
	  -I../rust/ignored main.cpp \
	  -o build/main

run_python: build/module$(EXTENSION_SUFFIX)
	PYTHONPATH=build python3 main.py

build/module$(EXTENSION_SUFFIX): ../rust/ignored/simulator.cpp wrapper.cpp module.cpp
	mkdir -p build
	clang++ -O3 \
	  -I../rust/ignored -I../extern/pybind11/include $(PYTHON_INCLUDES) \
	  module.cpp \
	  -shared -fPIC -o build/module$(EXTENSION_SUFFIX)
