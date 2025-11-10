# Create 2×2 bar plots with matplotlib (no seaborn, no custom colors).
# Each subplot compares models for a metric, with a dashed horizontal line for the y_wt baseline when available.
#%%
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Data parsed from the LaTeX table
# -----------------------------
models = ["DeepDTA", "AttentionDTA", "GraphDTA", "DGraphDTA", "MGraphDTA", "FDA", "Boltz-2"]
x = np.arange(len(models))
bar_width = 0.35

# (a) Same-ligand, different-modifications
a_mse_ywt = [0.61]*len(models)
a_mse_yhatwt = [0.63, 0.68, 0.73, 0.61, 0.61, 0.83, 0.81]
a_mse_yhat   = [0.62, 0.65, 1.07, 0.62, 0.59, 1.41, 0.79]
# Rp: only y_hat present; baseline y_wt not reported (dash line omitted)
a_rp_yhat    = [0.10, 0.10, -0.02, 0.03, 0.11, 0.18, 0.29]
a_rp_ywt = None  # not reported

# (b) Same-modification, different-ligands
b_mse_ywt = [0.53]*len(models)
b_mse_yhatwt = [0.58, 0.62, 0.90, 0.58, 0.57, 0.76, 0.76]
b_mse_yhat   = [0.56, 0.59, 1.26, 0.58, 0.54, 1.30, 0.75]
b_rp_ywt = [0.86]*len(models)  # baseline is reported and constant across models
b_rp_yhatwt = [0.84, 0.84, 0.79, 0.84, 0.84, 0.83, 0.78]
b_rp_yhat   = [0.84, 0.84, 0.70, 0.84, 0.85, 0.67, 0.79]

# -----------------------------
# Plotting
# -----------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

# Top-left: (a) MSE
ax = axes[0, 0]
ax.bar(x - bar_width/2, a_mse_yhatwt, width=bar_width, label=r'$\hat{y}_{WT}$', color='gray', alpha=0.2)
ax.bar(x + bar_width/2, a_mse_yhat,   width=bar_width, label=r'$\hat{y}$', color='red', alpha=0.5)
# Baseline: dashed y_wt
ax.axhline(y=a_mse_ywt[0], linestyle="--", linewidth=2.0, label=r'$y_{WT}$ baseline', color='black', alpha=0.7)
ax.set_title("Same-ligand, different-modifications", size=20)
ax.set_xticks(x)
ax.set_xticklabels([""] * len(models), rotation=20, ha="right")
ax.set_ylabel("MSE", size=15)
ax.set_ylim(0, 1.4)
ax.legend(loc="upper left", ncols=3, fontsize=9)
ax.grid(axis="y", linestyle=":", alpha=0.4)

# Top-right: (b) MSE
ax = axes[0, 1]
ax.bar(x - bar_width/2, b_mse_yhatwt, width=bar_width, label=r'$\hat{y}_{WT}$', color='gray', alpha=0.2)
ax.bar(x + bar_width/2, b_mse_yhat,   width=bar_width, label=r'$\hat{y}$', color='red', alpha=0.5)
ax.axhline(y=b_mse_ywt[0], linestyle="--", linewidth=2.0, label=r'$y_{WT}$ baseline', color='black', alpha=0.7)
ax.set_title("Same-modification, different-ligands", size=20)
ax.set_ylim(0, 1.4)
ax.set_xticks(x)
ax.set_xticklabels([""] * len(models), rotation=20, ha="right")
# ax.set_ylabel("MSE")
# ax.legend(loc="upper left", ncols=3, fontsize=9)
ax.grid(axis="y", linestyle=":", alpha=0.4)

# Bottom-left: (a) Rp
ax = axes[1, 0]
ax.bar(x - bar_width/2, a_rp_yhat, width=bar_width, label=r'$\hat{y}$', color='red', alpha=0.5)
# No y_wt baseline is reported here; annotate
#ax.text(0.02, 0.95, r'$y_{WT}$ baseline not reported', transform=ax.transAxes, va="top", ha="left", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=20, ha="right", size=15)
ax.set_ylabel(r"$R_p$", size=15)
ax.set_ylim(-0.05, 1.0)
# ax.legend(loc="upper left", fontsize=9)
ax.grid(axis="y", linestyle=":", alpha=0.4)

# Bottom-right: (b) Rp
ax = axes[1, 1]
ax.bar(x - bar_width/2, b_rp_yhatwt, width=bar_width, label=r'$\hat{y}_{WT}$', color='gray', alpha=0.2)
ax.bar(x + bar_width/2, b_rp_yhat,   width=bar_width, label=r'$\hat{y}$', color='red', alpha=0.5)
# Baseline y_wt as dashed line
ax.axhline(y=b_rp_ywt[0], linestyle="--", linewidth=2.0, label=r'$y_{WT}$ baseline', color='black', alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=20, ha="right", size=15)
# ax.set_ylabel(r"$R_p$")
ax.set_ylim(0.5, 1.0)
# ax.legend(loc="lower left", ncols=3, fontsize=9)
ax.grid(axis="y", linestyle=":", alpha=0.4)
ax.set_ylim(-0.05, 1.0)
fig.tight_layout()
fig.show()


# Save outputs
#out_png = "/mnt/data/wt_to_mod_generalization_2x2.png"
#out_pdf = "/mnt/data/wt_to_mod_generalization_2x2.pdf"
#plt.savefig(out_png, bbox_inches="tight", dpi=300)
#plt.savefig(out_pdf, bbox_inches="tight")





# %%
import numpy as np
import matplotlib.pyplot as plt

# ===== Data (means only; std in table omitted for plotting) =====
models = ["DeepDTA", "AttentionDTA", "GraphDTA", "DGraphDTA", "MGraphDTA", "FDA", "Boltz-2"]
x = np.arange(len(models))

# (a) Same-ligand, different-modifications
a_mse_ywt    = np.array([0.64]*len(models))
a_mse_yhatwt = np.array([0.63, 0.70, 0.71, 0.62, 0.62, 0.87, 0.79])
a_mse_yhat   = np.array([0.62, 0.68, 1.32, 0.63, 0.62, 1.28, 0.79])
a_mse_yhatft = np.array([0.33, 0.34, 0.83, 0.37, 0.43, 0.36, 0.42])

# Rp (a): only hat{y} and hat{y}_{FT} reported
a_rp_yhat   = np.array([-0.06, -0.03, -0.16, -0.03, -0.06,  0.20, 0.14])
a_rp_yhatft = np.array([ 0.17,  0.09,  0.02,  0.05, -0.04,  0.21, 0.20])

# (b) Same-modification, different-ligands
b_mse_ywt    = np.array([0.56]*len(models))
b_mse_yhatwt = np.array([0.63, 0.67, 1.10, 0.63, 0.59, 0.79, 0.74])
b_mse_yhat   = np.array([0.62, 0.62, 1.45, 0.64, 0.55, 1.15, 0.70])
b_mse_yhatft = np.array([0.54, 0.42, 1.20, 0.50, 0.45, 2.27, 0.62])

# Rp (b): all four present
b_rp_ywt    = np.array([0.78]*len(models))
b_rp_yhatwt = np.array([0.76, 0.77, 0.75, 0.78, 0.77, 0.75, 0.73])
b_rp_yhat   = np.array([0.76, 0.77, 0.66, 0.78, 0.78, 0.67, 0.76])
b_rp_yhatft = np.array([0.78, 0.80, 0.70, 0.78, 0.80, 0.56, 0.74])

# ===== Plotting (2x2) =====
fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
bar_width = 0.27

# --- Top-left: (a) MSE ---
ax = axes[0, 0]
ax.bar(x - bar_width,      a_mse_yhatwt, width=bar_width, label=r'$\hat{y}_{WT}$', alpha=0.2, color='gray')
ax.bar(x,                  a_mse_yhat,   width=bar_width, label=r'$\hat{y}$',     alpha=0.2, color='red')
ax.bar(x + bar_width,      a_mse_yhatft, width=bar_width, label=r'$\hat{y}_{FT}$',alpha=0.5, color='blue')
ax.axhline(y=a_mse_ywt[0], linestyle="--", linewidth=2.0, label=r'$y_{WT}$ baseline', color='black', alpha=0.8)
ax.set_title("Same-ligand, different-modifications", size=20)
ax.set_xticks(x); ax.set_xticklabels([""]*len(models))
ax.set_ylabel("MSE", size=15)
ax.set_ylim(0, 2.5)
ax.legend(loc="upper left", ncols=2, fontsize=9)
ax.grid(axis="y", linestyle=":", alpha=0.4)

# --- Top-right: (b) MSE ---
ax = axes[0, 1]
ax.bar(x - bar_width,      b_mse_yhatwt, width=bar_width, label=r'$\hat{y}_{WT}$', alpha=0.2, color='gray')
ax.bar(x,                  b_mse_yhat,   width=bar_width, label=r'$\hat{y}$',     alpha=0.2, color='red')
ax.bar(x + bar_width,      b_mse_yhatft, width=bar_width, label=r'$\hat{y}_{FT}$',alpha=0.5, color='blue')
ax.axhline(y=b_mse_ywt[0], linestyle="--", linewidth=2.0, label=r'$y_{WT}$ baseline', color='black', alpha=0.8)
ax.set_title("Same-modification, different-ligands", size=20)
ax.set_xticks(x); ax.set_xticklabels([""]*len(models))
ax.set_ylim(0, 2.5)
ax.grid(axis="y", linestyle=":", alpha=0.4)

# --- Bottom-left: (a) Rp ---
ax = axes[1, 0]
ax.bar(x - bar_width/2, a_rp_yhat,   width=bar_width, label=r'$\hat{y}$',     alpha=0.2, color='red')
ax.bar(x + bar_width/2, a_rp_yhatft, width=bar_width, label=r'$\hat{y}_{FT}$',alpha=0.5, color='blue')
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=20, ha="right", size=15)
ax.set_ylabel(r"$R_p$", size=15)
ax.set_ylim(-0.2, 1.0)
# ax.legend(loc="upper left", fontsize=9)
ax.grid(axis="y", linestyle=":", alpha=0.4)

# --- Bottom-right: (b) Rp ---
ax = axes[1, 1]
ax.bar(x - bar_width,      b_rp_yhatwt, width=bar_width, label=r'$\hat{y}_{WT}$', alpha=0.2, color='gray')
ax.bar(x,                  b_rp_yhat,   width=bar_width, label=r'$\hat{y}$',     alpha=0.2, color='red')
ax.bar(x + bar_width,      b_rp_yhatft, width=bar_width, label=r'$\hat{y}_{FT}$',alpha=0.5, color='blue')
ax.axhline(y=b_rp_ywt[0], linestyle="--", linewidth=2.0, label=r'$y_{WT}$ baseline', color='black', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=20, ha="right", size=15)
ax.set_ylim(-0.2, 1.0)
# ax.legend(loc="lower left", ncols=2, fontsize=9)
ax.grid(axis="y", linestyle=":", alpha=0.4)

#fig.suptitle("Few-shot Modification Generalization (Include $\\hat{y}_{FT}$)", y=1.02, fontsize=18)
fig.tight_layout()
plt.show()

# %%
