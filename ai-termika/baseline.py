"""
AI Termika — Baseline řešení (data-only MLP)
Jednoduchá MLP síť trénovaná pouze na měřených datech, bez fyzikální loss.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import os

# nastavení
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Používám: {device}")

torch.manual_seed(42)
np.random.seed(42)

# ============================================================
# 1. Načtení dat
# ============================================================
DATA_DIR = os.path.join(os.path.dirname(__file__), "ai_termika_dataset", "student")

train_df = pd.read_csv(f"{DATA_DIR}/train_measurements.csv")
boundary_df = pd.read_csv(f"{DATA_DIR}/boundary_partial.csv")
initial_df = pd.read_csv(f"{DATA_DIR}/initial_sparse.csv")
test_df = pd.read_csv(f"{DATA_DIR}/test_points.csv")

with open(f"{DATA_DIR}/constants.json") as f:
    constants = json.load(f)

print(f"Train bodů: {len(train_df)}, Boundary bodů: {len(boundary_df)}, Initial bodů: {len(initial_df)}")
print(f"Test bodů: {len(test_df)}")

# Spojíme všechna dostupná data (train + boundary + initial) do jednoho datasetu
all_data = pd.concat([train_df, boundary_df, initial_df], ignore_index=True)
print(f"Celkem trénovacích bodů: {len(all_data)}")


# Převod na tensory
def df_to_tensors(df, cols_x, col_y=None):
    X = torch.tensor(df[cols_x].values, dtype=torch.float32, device=device)
    if col_y:
        y = torch.tensor(df[col_y].values, dtype=torch.float32, device=device).unsqueeze(1)
        return X, y
    return X


X_all, y_all = df_to_tensors(all_data, ["x", "t"], "u")
X_test = df_to_tensors(test_df, ["x", "t"])


# ============================================================
# 2. Model — jednoduchá MLP
# ============================================================
class MLP(nn.Module):
    def __init__(self, hidden_size=64, n_layers=3):
        super().__init__()

        layers = []
        layers.append(nn.Linear(2, hidden_size))
        layers.append(nn.Tanh())

        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(hidden_size, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


model = MLP(hidden_size=64, n_layers=3).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Počet parametrů: {n_params:,}")


# ============================================================
# 3. Trénování — pouze data loss
# ============================================================
N_EPOCHS = 10000
LR = 1e-3

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3000, gamma=0.5)
mse = nn.MSELoss()

history = []

print("\nZačínám trénování...")

for epoch in range(N_EPOCHS):
    optimizer.zero_grad()

    u_pred = model(X_all)
    loss = mse(u_pred, y_all)

    loss.backward()
    optimizer.step()
    scheduler.step()

    history.append(loss.item())

    if (epoch + 1) % 1000 == 0:
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1:5d}/{N_EPOCHS} | Loss: {loss.item():.6f} | LR: {lr_now:.1e}")

print("\nTrénování dokončeno!")


# ============================================================
# 4. Generování submission.csv
# ============================================================
model.eval()
with torch.no_grad():
    u_pred_test = model(X_test).cpu().numpy().flatten()

submission = pd.DataFrame({
    "id": test_df["id"].values,
    "u": np.round(u_pred_test, 6),
})

OUTPUT_DIR = os.path.dirname(__file__)
submission.to_csv(os.path.join(OUTPUT_DIR, "submission_baseline.csv"), index=False)
print(f"\nUloženo {len(submission)} predikcí do submission_baseline.csv")

# Statistiky
print(f"Predikce — min: {u_pred_test.min():.4f}, max: {u_pred_test.max():.4f}, mean: {u_pred_test.mean():.4f}")
print(f"Train data — min: {all_data['u'].min():.4f}, max: {all_data['u'].max():.4f}, mean: {all_data['u'].mean():.4f}")


# ============================================================
# 5. Evaluace proti ground truth (pokud dostupný)
# ============================================================
gt_path = os.path.join(os.path.dirname(__file__), "ai_termika_dataset", "organizer", "test_ground_truth.csv")
if os.path.exists(gt_path):
    gt_df = pd.read_csv(gt_path)
    merged = submission.merge(gt_df, on="id", suffixes=("_pred", "_true"))
    rmse = np.sqrt(np.mean((merged["u_pred"] - merged["u_true"]) ** 2))
    print(f"\n=== RMSE na ground truth: {rmse:.6f} ===")
