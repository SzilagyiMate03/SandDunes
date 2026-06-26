import numpy as np
import matplotlib.pyplot as plt
import time

import dune_gpu   # a lefordított .so modul

# ── Elérhető GPU-k listázása ──────────────────────────────────
print(dune_gpu.list_devices())

# ── Szimuláció létrehozása ────────────────────────────────────
factor = 10
field = dune_gpu.DuneFieldGPU(
    nbx=3*4*factor, nby=8*4*factor,
    bx=8,   by=3,
    init_height=3,
    kernel_path="dune_gpu.cl",  # relatív útvonal
    platform_idx=0,
    device_idx=0,
)

# ── Futtatás ──────────────────────────────────────────────────
print("Futtatás...")
t0 = time.perf_counter()
H = field.runNsteps(2400)
dt = time.perf_counter() - t0
print(f"Runtime: {dt:.3f} s")

# ── Megjelenítés ──────────────────────────────────────────────
plt.figure(figsize=(8, 6))
plt.imshow(H, cmap="hot", origin="lower")
plt.colorbar(label="Magasság")
plt.title("Homokdűne szimuláció – GPU")
plt.xlabel("X")
plt.ylabel("Y")
plt.tight_layout()
plt.savefig("dune_result.png", dpi=300)
print("Kép mentve: dune_result.png")
