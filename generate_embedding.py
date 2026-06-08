"""
Regenerate app_artifacts/embedding_data.npz and app_artifacts/diagnostic_landscape.png
using the fitted VotingClassifier pipeline.
Run after retraining the model in NEW-finetuning.ipynb.
"""
import numpy as np
import pandas as pd
import pickle
import os
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

# --- Load artifacts ---
print("Loading model and metadata...")
with open("best_model_finetuned.pkl", "rb") as f:
    model = pickle.load(f)
with open("model_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)
with open("split_indices.pkl", "rb") as f:
    split_indices = pickle.load(f)

final_features = list(metadata["features"])
train_idx = split_indices["train_idx"]   # original DataFrame index values (not positional)
LABEL_COL = "GRUP"
G1, G2 = 1, 2

# --- Load raw data ---
print("Loading data...")
df = pd.read_excel("Miyokardit_08.12.xlsx", sheet_name=0)
df = df.dropna(subset=[LABEL_COL])

# --- Extract training split using label-based indexing (.loc) ---
df_train = df.loc[train_idx].copy()
X_train = df_train[final_features].copy()
y_train = df_train[LABEL_COL].values

# --- Drop rows with any NaN ---
mask = ~X_train.isna().any(axis=1)
X_train = X_train[mask]
y_train = y_train[mask]
print(f"Training samples after NaN drop: {len(X_train)}")
print(f"Features: {len(final_features)}")

# --- Transform through fitted UncertaintyTransformer ---
print("Transforming through UncertaintyTransformer...")
X_unc = model.named_steps["uncertainty"].transform(X_train)
unc_feature_names = list(model.named_steps["uncertainty"].feature_names_in_)
X_unc_df = pd.DataFrame(X_unc, columns=unc_feature_names)

# --- Scale with fresh StandardScaler (fit on DataFrame so feature_names_in_ is set) ---
print("Scaling with fresh StandardScaler...")
tsne_scaler = StandardScaler()
X_std = tsne_scaler.fit_transform(X_unc_df)

# --- Run t-SNE with exact same parameters as NEW_uncertainty.ipynb ---
print("Running t-SNE (this may take a minute)...")
tsne = TSNE(n_components=2, perplexity=50, learning_rate="auto",
            init="pca", random_state=42)
X_emb = tsne.fit_transform(X_std)
print("t-SNE done.")

# --- Save embedding and scaler ---
os.makedirs("app_artifacts", exist_ok=True)
out_path = os.path.join("app_artifacts", "embedding_data.npz")
np.savez(out_path, X_std=X_std, X_emb=X_emb, y=y_train)
print(f"Saved embedding to {out_path}")

scaler_path = os.path.join("app_artifacts", "tsne_scaler.pkl")
with open(scaler_path, "wb") as f:
    pickle.dump(tsne_scaler, f)
print(f"Saved tsne_scaler to {scaler_path}")

# --- Generate diagnostic landscape PNG (exact notebook logic) ---
print("Generating diagnostic landscape PNG...")
labels = y_train.copy()
pad = 2.0
xmin, xmax = X_emb[:, 0].min() - pad, X_emb[:, 0].max() + pad
ymin, ymax = X_emb[:, 1].min() - pad, X_emb[:, 1].max() + pad
resolution = 1000
xs = np.linspace(xmin, xmax, resolution)
ys = np.linspace(ymin, ymax, resolution)
xx, yy = np.meshgrid(xs, ys)
grid = np.vstack([xx.ravel(), yy.ravel()])

class1 = X_emb[labels == G1]
class2 = X_emb[labels == G2]
kde1 = gaussian_kde(class1.T, bw_method="scott")
kde2 = gaussian_kde(class2.T, bw_method="scott")
z1 = kde1(grid).reshape(xx.shape)
z2 = kde2(grid).reshape(xx.shape)

q = 0.6
level1 = np.quantile(z1, q)
level2 = np.quantile(z2, q)

def normalise(z, clip=0.98):
    zmax = np.quantile(z, clip)
    return np.clip(z / zmax, 0, 1)

alpha1 = normalise(z1) ** 0.5
alpha2 = normalise(z2) ** 0.5
alpha1[z1 < level1] = 0.0
alpha2[z2 < level2] = 0.0

overlap_mask = (alpha1 > 0) & (alpha2 > 0)
alpha_overlap = np.maximum(alpha1, alpha2)
alpha_overlap[~overlap_mask] = 0.0
alpha1[overlap_mask] = 0.0
alpha2[overlap_mask] = 0.0

eps = 1e-12
total = z1 + z2 + eps
t = (z1 - z2) / total
shift_strength = 0.3

R = np.full_like(t, 0.5)
G_arr = np.full_like(t, 0.5)
B = np.full_like(t, 0.5)
pos = t > 0
R[pos] += shift_strength * t[pos]
G_arr[pos] -= shift_strength * t[pos]
B[pos] -= shift_strength * t[pos]
neg = t < 0
B[neg] += shift_strength * (-t[neg])
R[neg] -= shift_strength * (-t[neg])
G_arr[neg] -= shift_strength * (-t[neg])

shape = (*alpha1.shape, 4)
red_img = np.zeros(shape); red_img[..., 0] = 1.0; red_img[..., 3] = alpha1
blue_img = np.zeros(shape); blue_img[..., 2] = 1.0; blue_img[..., 3] = alpha2
shape = (*alpha_overlap.shape, 4)
over_img = np.zeros(shape)
over_img[..., 0] = R; over_img[..., 1] = G_arr; over_img[..., 2] = B
over_img[..., 3] = alpha_overlap

fig, ax = plt.subplots()
fig.patch.set_facecolor("white")
for img in [over_img, red_img, blue_img]:
    ax.imshow(img, extent=(xmin, xmax, ymin, ymax), origin="lower", interpolation="bilinear")
ax.set_axis_off()
legend_handles = [
    Patch(facecolor="red",  edgecolor="red",  label="Myocarditis"),
    Patch(facecolor="blue", edgecolor="blue", label="ACS"),
    Patch(facecolor="gray", edgecolor="gray", label="Uncertain"),
]
ax.legend(handles=legend_handles, loc="upper right", frameon=False)
fig.tight_layout()

png_path = os.path.join("app_artifacts", "diagnostic_landscape.png")
plt.savefig(png_path, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved diagnostic landscape PNG to {png_path}")
