# ===== Compiler =====
CXX := g++
PYTHON := python3

# ===== Pybind11 includes =====
PYBIND11_INC := $(shell $(PYTHON) -m pybind11 --includes)

# ===== Python extension suffix (platform dependent!) =====
EXT_SUFFIX := $(shell $(PYTHON)-config --extension-suffix)

# ===== Flags =====
CXXFLAGS := -O3 -Wall -shared -std=c++17 -fPIC -fopenmp $(PYBIND11_INC)

# ===== Target =====
TARGET := dune$(EXT_SUFFIX)
SRC := sand.cpp

# ===== Build =====
$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET)

# ===== Clean =====
clean:
	rm -f $(TARGET)
