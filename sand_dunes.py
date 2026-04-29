import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import dune

# ===== PARAMÉTEREK =====
NBx, NBy = 128, 128
Bx, By   = 8, 3
init_height = 10
Nx, Ny   = NBx*Bx, NBy*By

N_steps = 2000
steps_per_frame = Bx*By   # ennyi lépésenként mentünk egy képkockát

output_file = "dune_simulation.mp4"

# ===== MODELL =====
df = dune.DuneField(NBx, NBy, Bx, By, init_height)

# ===== FIGURE =====
fig = plt.figure(figsize=(16, 9))

gs = fig.add_gridspec(1, 2, width_ratios=[50, 1], wspace=0.02)
ax = fig.add_subplot(gs[0])
cax = fig.add_subplot(gs[1])

H0 = df.runNsteps(0)

im = ax.imshow(H0, cmap="plasma_r", origin="lower", aspect="auto" )
im.cmap.set_bad(color='black')
cb = plt.colorbar(im, cax=cax)

cb.ax.tick_params(labelsize=12)  # kisebb számok
title = ax.set_title("Step: 0", fontsize=14)
fig.subplots_adjust(left=0.04, right=0.94, bottom=0.04, top=0.94)


# ===== FRAME GENERATOR =====
def update(frame):
    global df
    H = df.runNsteps(steps_per_frame)
    H_masked = np.ma.masked_where(H == 0, H)
    im.set_data(H_masked)
    im.set_clim(0, 100)
    title.set_text(f"Step: {frame * steps_per_frame}")
    # ===== PROGRESS PRINT =====
    if frame % 10 == 0:
        print(f"\rProgress: {100*frame/N_steps:.1f}%", end="", flush=True)
    return [im]

ani = animation.FuncAnimation(fig, update, frames=N_steps, blit=False)

# ===== VIDEÓ MENTÉS =====
Writer = animation.writers['ffmpeg']
writer = Writer(fps=25, bitrate=20000)

ani.save(output_file, writer=writer)

print(f"\nSaved video to {output_file}")
