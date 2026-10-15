import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Ensure figures dir exists
os.makedirs("paper/figures", exist_ok=True)

# Set IEEE compliant typography
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.titlesize': 8,
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})

# -------------------------------------------------------------
# 1. STATION LOCATION MAP
# -------------------------------------------------------------
df = pd.read_csv("data/station.csv")

fig, ax = plt.subplots(figsize=(3.4, 2.8), dpi=300)
sc = ax.scatter(df['longitude'], df['latitude'], c='#1f77b4', edgecolors='#0d3b66', s=35, zorder=3, label='Monitoring Station')

# Label a representative subset to avoid unreadable overlapping
for i, row in df.iterrows():
    # Only annotate some prominent stations or offsets
    short_name = row['station'].replace('_aq', '')
    if i % 3 == 0:
        ax.annotate(short_name, (row['longitude'], row['latitude']), 
                    textcoords="offset points", xytext=(4, 2), fontsize=5.5, alpha=0.85)

ax.set_xlabel("Longitude (degrees East)")
ax.set_ylabel("Latitude (degrees North)")
ax.grid(True, linestyle='--', alpha=0.4, zorder=1)
ax.set_title("Spatial Distribution of 35 Air Quality Stations (Beijing)", fontsize=8, pad=6)
ax.legend(loc='lower right', framealpha=0.9, fontsize=6.5)

plt.tight_layout()
plt.savefig("paper/figures/station_map.pdf", format='pdf', bbox_inches='tight')
plt.savefig("paper/figures/station_map.png", format='png', dpi=300, bbox_inches='tight')
plt.close()
print("Station map saved.")

# -------------------------------------------------------------
# 2. ARCHITECTURE DIAGRAM (AtmoFuseNet)
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.0, 3.2), dpi=300)
ax.axis('off')
ax.set_xlim(0, 14)
ax.set_ylim(0, 6.5)

# Style parameters
box_color = '#e8f4f8'
edge_color = '#2b5c8f'
box_ode = '#fff2df'
edge_ode = '#d97706'
box_fusion = '#f3e8ff'
edge_fusion = '#7e22ce'

def draw_box(x, y, w, h, title, subtitle="", bg='#f0f4f8', border='#334155', title_color='#0f172a', title_size=7.5, sub_size=6.0):
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.15",
                                  linewidth=1.0, edgecolor=border, facecolor=bg, zorder=2)
    ax.add_patch(rect)
    if subtitle:
        ax.text(x + w/2, y + h*0.62, title, ha='center', va='center', fontsize=title_size, weight='bold', color=title_color, zorder=3)
        ax.text(x + w/2, y + h*0.28, subtitle, ha='center', va='center', fontsize=sub_size, color='#475569', zorder=3)
    else:
        ax.text(x + w/2, y + h/2, title, ha='center', va='center', fontsize=title_size, weight='bold', color=title_color, zorder=3)

def draw_arrow(x1, y1, x2, y2, label=""):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="#334155", lw=1.0, shrinkA=2, shrinkB=2), zorder=4)
    if label:
        ax.text((x1+x2)/2, (y1+y2)/2 + 0.15, label, ha='center', va='bottom', fontsize=5.8, color='#334155', zorder=4)

# 1. Inputs
draw_box(0.2, 2.5, 1.8, 1.5, "Input Graph Series", "24 Hours (6 Variables)\nPM2.5, Temp, Press, Wind", bg='#f1f5f9', border='#64748b')

# Arrow 1
draw_arrow(2.0, 3.25, 2.5, 3.25, "PM2.5")

# 2. GRU Encoder
draw_box(2.5, 2.5, 1.8, 1.5, "GRU Encoder", "Hidden Units (64)\nPer-station recurrent", bg='#e0f2fe', border='#0284c7')

# Arrow 2
draw_arrow(4.3, 3.25, 4.8, 3.25)

# 3. Latent Reparameterization
draw_box(4.8, 2.3, 1.8, 1.9, "Latent Projector", "MLP: 64 -> 50 -> 8\nMean & Std (Dim 4)\nSampling z0 (3 Traj)", bg='#ecfdf5', border='#059669')

# Arrow 3 (splits into ODE)
draw_arrow(6.6, 3.25, 7.2, 4.2)
draw_arrow(6.6, 3.25, 7.2, 2.2)

# Large ODE Container Box
ode_container = patches.FancyBboxPatch((7.0, 0.8), 4.3, 4.8, boxstyle="round,pad=0.1,rounding_size=0.2",
                                      linewidth=1.2, linestyle='--', edgecolor='#ea580c', facecolor='#fffaf5', zorder=1)
ax.add_patch(ode_container)
ax.text(9.15, 5.35, "Neural ODE with Physics Guidance (Dormand-Prince dopri5)", ha='center', va='center', fontsize=7.5, weight='bold', color='#c2410c')

# Diffusion Branch inside ODE
draw_box(7.2, 3.6, 1.8, 1.3, "Diffusion Branch", "Scaled Laplacian (K=2)\nChebyshev GCN\nParam LayerParams", bg='#fef3c7', border='#d97706')

# Advection Branch inside ODE
draw_box(7.2, 1.4, 1.8, 1.5, "Advection Branch", "Wind Vars -> Linear(2,1)\nDynamic Edge Difference\nDirected Laplacian GCN", bg='#ffedd5', border='#ea580c')

# Wind arrow from input
ax.annotate('', xy=(7.2, 1.8), xytext=(2.0, 2.7),
            arrowprops=dict(arrowstyle="->", color="#94a3b8", lw=0.9, linestyle=":", shrinkA=2, shrinkB=2), zorder=4)
ax.text(4.2, 1.9, "Wind Speed & Direction", ha='center', va='bottom', fontsize=5.5, color='#64748b', style='italic', zorder=4)

# Arrows to Gated Fusion
draw_arrow(9.0, 4.25, 9.6, 3.5, "H_diff")
draw_arrow(9.0, 2.15, 9.6, 2.9, "H_adv")

# Gated Fusion Box
draw_box(9.6, 2.5, 1.5, 1.5, "Gated Fusion", "H = z*H_diff + (1-z)*H_adv\nz = Sigmoid(W_diff+W_adv)", bg='#f3e8ff', border='#7e22ce')

# Arrow from ODE container to Decoder
draw_arrow(11.3, 3.25, 11.8, 3.25, "Trajectories")

# Decoder
draw_box(11.8, 2.5, 1.8, 1.5, "Decoder", "Parameter-free Average\nAcross Latent & Traj\nMulti-Step PM2.5 (24h)", bg='#f1f5f9', border='#475569')

plt.tight_layout()
plt.savefig("paper/figures/atmofusenet_architecture.pdf", format='pdf', bbox_inches='tight')
plt.savefig("paper/figures/atmofusenet_architecture.png", format='png', dpi=300, bbox_inches='tight')
plt.close()
print("Architecture figure saved.")
