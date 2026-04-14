"""
AI Termika — PINN řešení (reference solution)
Physics-Informed Neural Network pro predikci šíření tepla v kovové tyči.
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

alpha = constants["alpha"]
print(f"Přibližné alpha: {alpha}")
print(f"Train bodů: {len(train_df)}, Boundary bodů: {len(boundary_df)}, Initial bodů: {len(initial_df)}")
print(f"Test bodů: {len(test_df)}")


# Převod na tensory
def df_to_tensors(df, cols_x, col_y=None):
    X = torch.tensor(df[cols_x].values, dtype=torch.float32, device=device)
    if col_y:
        y = torch.tensor(df[col_y].values, dtype=torch.float32, device=device).unsqueeze(1)
        return X, y
    return X


X_train, y_train = df_to_tensors(train_df, ["x", "t"], "u")
X_boundary, y_boundary = df_to_tensors(boundary_df, ["x", "t"], "u")
X_initial, y_initial = df_to_tensors(initial_df, ["x", "t"], "u")
X_test = df_to_tensors(test_df, ["x", "t"])


# ============================================================
# 2. Model
# ============================================================
class PINN(nn.Module):
    def __init__(self, hidden_size=128, n_layers=5):
        super().__init__()

        layers = []
        layers.append(nn.Linear(2, hidden_size))
        layers.append(nn.Tanh())

        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(hidden_size, 1))

        self.net = nn.Sequential(*layers)

        # Xavier inicializace
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


model = PINN(hidden_size=128, n_layers=5).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Počet parametrů: {n_params:,}")


# ============================================================
# 3. Physics loss
# ============================================================
def physics_residual(model, x_col, alpha_val):
    """Residual rovnice vedení tepla: du/dt - alpha * d²u/dx²."""
    x_col = x_col.requires_grad_(True)
    u = model(x_col)

    grad_u = torch.autograd.grad(
        u, x_col,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
    )[0]

    du_dx = grad_u[:, 0:1]
    du_dt = grad_u[:, 1:2]

    d2u_dx2 = torch.autograd.grad(
        du_dx, x_col,
        grad_outputs=torch.ones_like(du_dx),
        create_graph=True,
    )[0][:, 0:1]

    return du_dt - alpha_val * d2u_dx2


def sample_collocation_points(n_points, device):
    x = torch.rand(n_points, 1, device=device)
    t = torch.rand(n_points, 1, device=device)
    return torch.cat([x, t], dim=1)


# ============================================================
# 4. Trénování
# ============================================================
N_EPOCHS = 15000
LR = 1e-3
N_COLLOCATION = 2000

LAMBDA_DATA = 1.0
LAMBDA_BOUNDARY = 5.0
LAMBDA_INITIAL = 5.0
LAMBDA_PHYSICS = 0.5

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)

mse = nn.MSELoss()

print("\nZačínám trénování...")

for epoch in range(N_EPOCHS):
    optimizer.zero_grad()

    # Data loss
    u_pred_train = model(X_train)
    loss_data = mse(u_pred_train, y_train)

    # Boundary loss
    u_pred_boundary = model(X_boundary)
    loss_boundary = mse(u_pred_boundary, y_boundary)

    # Initial condition loss
    u_pred_initial = model(X_initial)
    loss_initial = mse(u_pred_initial, y_initial)

    # Physics loss
    x_col = sample_collocation_points(N_COLLOCATION, device)
    residual = physics_residual(model, x_col, alpha)
    loss_physics = torch.mean(residual ** 2)

    # Celková loss
    loss_total = (
        LAMBDA_DATA * loss_data
        + LAMBDA_BOUNDARY * loss_boundary
        + LAMBDA_INITIAL * loss_initial
        + LAMBDA_PHYSICS * loss_physics
    )

    loss_total.backward()
    optimizer.step()
    scheduler.step()

    if (epoch + 1) % 1000 == 0:
        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1:5d}/{N_EPOCHS} | "
            f"Total: {loss_total.item():.6f} | "
            f"Data: {loss_data.item():.6f} | "
            f"Boundary: {loss_boundary.item():.6f} | "
            f"Initial: {loss_initial.item():.6f} | "
            f"Physics: {loss_physics.item():.6f} | "
            f"LR: {lr_now:.1e}"
        )

print("\nTrénování dokončeno!")


# ============================================================
# 5. Generování submission.csv
# ============================================================
model.eval()
with torch.no_grad():
    u_pred_test = model(X_test).cpu().numpy().flatten()

submission = pd.DataFrame({
    "id": test_df["id"].values,
    "u": np.round(u_pred_test, 6),
})

OUTPUT_DIR = os.path.dirname(__file__)
submission.to_csv(os.path.join(OUTPUT_DIR, "submission.csv"), index=False)
print(f"\nUloženo {len(submission)} predikcí do submission.csv")

# Statistiky
print(f"Predikce — min: {u_pred_test.min():.4f}, max: {u_pred_test.max():.4f}, mean: {u_pred_test.mean():.4f}")
print(f"Train data — min: {train_df['u'].min():.4f}, max: {train_df['u'].max():.4f}, mean: {train_df['u'].mean():.4f}")


# ============================================================
# 6. Evaluace proti ground truth (pokud dostupný)
# ============================================================
gt_path = os.path.join(os.path.dirname(__file__), "ai_termika_dataset", "organizer", "test_ground_truth.csv")
if os.path.exists(gt_path):
    gt_df = pd.read_csv(gt_path)
    merged = submission.merge(gt_df, on="id", suffixes=("_pred", "_true"))
    rmse = np.sqrt(np.mean((merged["u_pred"] - merged["u_true"]) ** 2))
    print(f"\n=== RMSE na ground truth: {rmse:.6f} ===")
