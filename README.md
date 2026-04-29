# Dune Field Simulation (C++ + Pybind11 + Python)

## 📌 Overview

This project implements a parallel cellular automaton model for aeolian dune formation.
The core simulation is written in C++ (with OpenMP), exposed to Python via pybind11, and visualized using Python (NumPy + Matplotlib).

---

## ⚙️ Dependencies

### 🔹 System requirements

* C++17 compatible compiler (e.g. `g++`)
* Python ≥ 3.8
* OpenMP support (`-fopenmp`)
* ffmpeg (for video export)

### 🔹 Python packages

Install with:

```
pip install numpy matplotlib pybind11
```

```
sudo apt install ffmpeg
```

---

## 🛠️ Build Instructions

Compile the Python module:

```
make
```

This creates a shared library:

```
dune.cpython-XXX.so
```

---

## 🚀 Running the Simulation

Run the Python script:

```
python3 sand_dunes.py
```

The output is an `.mp4` video of the evolving dune field.

---

## 📐 Model Parameters

### C++ / Python interface

```python
df = dune.DuneField(nbx, nby, bx, by, init_height)
```

| Parameter     | Meaning                               |
| ------------- | ------------------------------------- |
| `nbx`, `nby`  | Number of blocks in x and y direction |
| `bx`, `by`    | Block size                            |
| `init_height` | Initial sand height (with ±1 noise)   |

Derived:

```
Nx = nbx * bx
Ny = nby * by
```

---

### Simulation parameters (Python side)

| Parameter         | Meaning                    |
| ----------------- | -------------------------- |
| `N_steps`         | Number of animation frames |
| `steps_per_frame` | Simulation steps per frame |
| `fps`             | Video frame rate           |
| `bitrate`         | Video quality              |

---

### Internal model parameters (C++)

| Parameter             | Value                         | Meaning                   |
| --------------------- | ----------------------------- | ------------------------- |
| `hopLength`           | 5                             | Saltation hop distance    |
| `avalancheThreshold`  | 2                             | Slope stability threshold |
| `tanTheta`            | ~0.804                        | Shadow angle              |
| `shadowCheckDistance` | ~1.5 * sqrt(Nx * init_height) | Max shadow range          |

---

## 🧠 Model Description

Each timestep consists of:

1. Random grain pickup (if not in shadow)
2. Saltation (downwind transport)
3. Probabilistic deposition
4. Avalanche relaxation (local slope stabilization)

Parallelization:

* Block-based domain decomposition
* OpenMP parallel loops
* Atomic updates for thread safety


## 📁 Output

* Video file: `dune_simulation.mp4`
* Resolution depends on grid size

---
