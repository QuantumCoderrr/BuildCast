import os, joblib, warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt 

# LightGBM baseline
import lightgbm as lgb

# PyTorch model
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------------- CONFIG ----------------
DATA_PATH = "Building_Dataset_2.xlsx"
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ARTIFACT_DIR = "./artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# -------------- LOAD DATA ---------------
df = pd.read_excel(DATA_PATH)
print("Loaded dataframe shape:", df.shape)
print("Columns:", df.columns.tolist())

target_col = "Energy_Consumption"
feature_cols = [c for c in df.columns if c != target_col]

# numeric_cols = ["Square Footage", "Number of Occupants", "Appliances Used", "Average Temperature"]
numeric_cols=["Square Footage",	"Building_Area",	"Floors",	"Year_Built", "Water_Consumption",	"CO2_Emissions",	"Number of Occupants",	"Appliances Used",	"Average Temperature"]
cat_cols = ["Building Type", "Day of Week"]

df_f = df.copy()
for c in numeric_cols:
    df_f[c] = df_f[c].fillna(df_f[c].median())
for c in cat_cols:
    df_f[c] = df_f[c].fillna("__missing__").astype(str)

# Encoding
ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
cat_arr = ohe.fit_transform(df_f[cat_cols])

scaler = StandardScaler()
num_arr = scaler.fit_transform(df_f[numeric_cols])

X = np.hstack([num_arr, cat_arr]).astype(np.float32)
y = df_f[target_col].values.astype(np.float32).reshape(-1,1)

print("Feature matrix shape:", X.shape, "Target shape:", y.shape)

joblib.dump(ohe, os.path.join(ARTIFACT_DIR, "ohe.pkl"))
joblib.dump(scaler, os.path.join(ARTIFACT_DIR, "scaler.pkl"))

# -------------- TRAIN/VAL/TEST SPLIT ---------------
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.25, random_state=RANDOM_SEED)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=RANDOM_SEED)

print("Train/val/test sizes:", len(X_train), len(X_val), len(X_test))

# ---------------- LightGBM baseline ----------------
lgb_train = lgb.Dataset(X_train, label=y_train.ravel())
lgb_val = lgb.Dataset(X_val, label=y_val.ravel(), reference=lgb_train)

params = {
    'objective': 'regression',
    'metric': 'rmse',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'seed': RANDOM_SEED,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'feature_fraction': 0.8
}
print("Training LightGBM baseline...")
bst = lgb.train(
    params,
    lgb_train,
    valid_sets=[lgb_train, lgb_val],
    num_boost_round=1000,
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=50)
    ]
)

y_pred_val = bst.predict(X_val, num_iteration=bst.best_iteration)
print("LightGBM val MAE:", mean_absolute_error(y_val, y_pred_val))
joblib.dump(bst, os.path.join(ARTIFACT_DIR, "lgb_baseline.pkl"))

# ---------------- PyTorch MLP (Advanced model) ----------------
class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(TabularDataset(X_train,y_train), batch_size=64, shuffle=True)
val_loader = DataLoader(TabularDataset(X_val,y_val), batch_size=64, shuffle=False)
test_loader = DataLoader(TabularDataset(X_test,y_test), batch_size=64, shuffle=False)

input_dim = X.shape[1]
class MLPAdvanced(nn.Module):
    def __init__(self, in_dim, hidden_dims=[256,128], dropout=0.2):
        super().__init__()
        layers = []
        cur = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(cur, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            cur = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(cur, cur//2),
            nn.ReLU(),
            nn.Linear(cur//2, 1)
        )
    def forward(self, x):
        h = self.body(x)
        return self.head(h)

model = MLPAdvanced(input_dim).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
loss_fn = nn.MSELoss()

def train_epoch(model, loader, opt):
    model.train()
    total_loss = 0
    for xb, yb in loader:
        xb = xb.to(DEVICE); yb = yb.to(DEVICE)
        opt.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        opt.step()
        total_loss += loss.item()*xb.size(0)
    return total_loss / len(loader.dataset)

def eval_model(model, loader):
    model.eval()
    preds = []; trues = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            pred = model(xb).cpu().numpy()
            preds.append(pred)
            trues.append(yb.numpy())
    preds = np.vstack(preds).ravel()
    trues = np.vstack(trues).ravel()
    return preds, trues

# Train loop
EPOCHS = 20
best_val_mae = 1e9
for ep in range(1, EPOCHS+1):
    tr_loss = train_epoch(model, train_loader, optimizer)
    val_preds, val_trues = eval_model(model, val_loader)
    val_mae = mean_absolute_error(val_trues, val_preds)
    if val_mae < best_val_mae:
        best_val_mae = val_mae
        torch.save(model.state_dict(), os.path.join(ARTIFACT_DIR, "mlp_advanced_best.pth"))
    if ep % 5 == 0 or ep==1:
        print(f"Epoch {ep}/{EPOCHS} - train loss {tr_loss:.6f}  val MAE {val_mae:.4f}")

# Load best and evaluate on test
model.load_state_dict(torch.load(os.path.join(ARTIFACT_DIR, "mlp_advanced_best.pth"), map_location=DEVICE))
test_preds, test_trues = eval_model(model, test_loader)
mae = mean_absolute_error(test_trues, test_preds)
mse = mean_squared_error(test_trues, test_preds)
rmse = np.sqrt(mse)  # ✅ Manual RMSE (works everywhere)
r2 = r2_score(test_trues, test_preds)

print("MLP Test MAE:", mae)
print("MLP Test RMSE:", rmse)
print("MLP Test R2:", r2)

torch.save(model.state_dict(), os.path.join(ARTIFACT_DIR, "mlp_advanced_final.pth"))
print("\n--- READY ---")
print("LightGBM saved to", os.path.join(ARTIFACT_DIR, "lgb_baseline.pkl"))
print("Encoders saved to", os.path.join(ARTIFACT_DIR, "ohe.pkl"), os.path.join(ARTIFACT_DIR, "scaler.pkl"))
print("MLP saved to", os.path.join(ARTIFACT_DIR, "mlp_advanced_final.pth"))

# ---------------- VISUALIZATION ----------------
plt.figure(figsize=(6,6))
plt.scatter(test_trues, test_preds, alpha=0.5)
plt.plot([test_trues.min(), test_trues.max()], [test_trues.min(), test_trues.max()], 'r--')
plt.xlabel("Actual Energy Consumption")
plt.ylabel("Predicted Energy Consumption")
plt.title("Actual vs Predicted")
plt.show()

plt.figure(figsize=(10,5))
plt.plot(test_trues[:100], label="Actual")
plt.plot(test_preds[:100], label="Predicted")
plt.legend()
plt.title("Energy Consumption (first 100 samples)")
plt.xlabel("Sample Index")
plt.ylabel("Energy Consumption")
plt.show()

residuals = test_trues - test_preds
plt.figure(figsize=(8,5))
plt.hist(residuals, bins=30, alpha=0.7)
plt.title("Residuals Distribution (Actual - Predicted)")
plt.xlabel("Residual")
plt.ylabel("Frequency")

plt.show() 
