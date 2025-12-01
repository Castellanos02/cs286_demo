WINDOW_SIZE = 32
feature_cols = ['latitude', 'longitude', 'dt_s']

X_list = []
y_list = []

for taxi_id, group in all_taxi.groupby('taxi_id'):
    g = group.sort_values('timestamp')
    feats = g[feature_cols].values.astype(np.float32)   # (N, 3)
    labels = g['label'].values.astype(np.int64)         # (N,)

    if len(g) < WINDOW_SIZE:
        continue  # skip very short trajectories

    # Per-taxi normalization (helps a lot)
    mean = feats.mean(axis=0, keepdims=True)
    std = feats.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    feats = (feats - mean) / std

    # Sliding windows
    for i in range(len(g) - WINDOW_SIZE + 1):
        x_win = feats[i:i + WINDOW_SIZE]          # (T, F)
        y_win = labels[i + WINDOW_SIZE - 1]       # label at last step
        X_list.append(x_win)
        y_list.append(y_win)

X = np.stack(X_list, axis=0)  # (num_samples, T, F)
y = np.array(y_list)          # (num_samples,)

print("Windowed X shape:", X.shape)  # (N, 32, 3)
print("Labels shape:", y.shape)
print("Class balance:", np.bincount(y))

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class TaxiWindowDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)    # (N, T, F)
        self.y = torch.from_numpy(y)    # (N,)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

train_ds = TaxiWindowDataset(X_train, y_train)
val_ds   = TaxiWindowDataset(X_val,   y_val)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False)

import torch.nn as nn

class CNNTravelState(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # -> (B, C, 1)
        )
        self.fc = nn.Linear(hidden_channels * 2, 1)  # binary output

    def forward(self, x):
        # x: (B, T, F) -> (B, F, T) for Conv1d
        x = x.transpose(1, 2)
        x = self.conv(x)        # (B, C, 1)
        x = x.squeeze(-1)       # (B, C)
        logits = self.fc(x)     # (B, 1)
        return logits.squeeze(-1)  # (B,)

import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNTravelState(in_channels=3, hidden_channels=32).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def run_epoch(loader, training=True):
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total = 0
    correct = 0

    with torch.set_grad_enabled(training):
        for xb, yb in loader:
            xb = xb.to(device)                 # (B, T, F)
            yb = yb.float().to(device)         # (B,)

            if training:
                optimizer.zero_grad()

            logits = model(xb)                 # (B,)
            loss = criterion(logits, yb)

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * xb.size(0)
            total += xb.size(0)

            preds = (torch.sigmoid(logits) > 0.5).long()
            correct += (preds.cpu() == yb.long().cpu()).sum().item()

    return total_loss / total, correct / total

EPOCHS = 10
for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = run_epoch(train_loader, training=True)
    val_loss, val_acc = run_epoch(val_loader, training=False)
    print(f"Epoch {epoch:02d} | "
          f"train loss {train_loss:.4f}, acc {train_acc:.3f} | "
          f"val loss {val_loss:.4f}, acc {val_acc:.3f}")

torch.save(model.state_dict(), "cnn_taxi_pretrained.pt")


# 0) Ensure timestamp is datetime and sorted
interploated_df['timestamp'] = pd.to_datetime(interploated_df['timestamp'])
interploated_df = interploated_df.sort_values('timestamp')

# 1) Label: moving = 1, stationary = 0
interploated_df['label'] = (interploated_df['travelstate'] == 'moving').astype(np.int64)

# 2) Time delta dt_s (seconds) between consecutive samples
interploated_df['t_prev'] = interploated_df['timestamp'].shift(1)
interploated_df['dt_s'] = (interploated_df['timestamp'] - interploated_df['t_prev']).dt.total_seconds()
interploated_df['dt_s'] = interploated_df['dt_s'].fillna(0.0)
interploated_df['dt_s'] = interploated_df['dt_s'].clip(lower=0.0, upper=3600.0)

# 3) Features we’ll use for the CNN
feature_cols = ['latitude', 'longitude', 'dt_s']

# 4) ***Important: day-only (no time)***
interploated_df['day'] = interploated_df['timestamp'].dt.date   # <--- Python date objects

import numpy as np

WINDOW_SIZE = 32
feature_cols = ['latitude', 'longitude', 'dt_s']

X_list_B = []
y_list_B = []
event_list_B = []      # day per window
subject_list_B = []    # subject per window

# Drop rows with missing features
clean_df = interploated_df.dropna(subset=feature_cols)

# Build windows PER SUBJECT so they don't cross users
for sid, group in clean_df.groupby("subject_id"):
    g = group.sort_values("timestamp")

    feats    = g[feature_cols].values.astype(np.float32)   # (N_s, 3)
    labels   = g['label'].values.astype(np.int64)          # (N_s,)
    days     = g['day'].values                             # (N_s,)
    subjects = g['subject_id'].values                      # (N_s,) (all = sid)

    if len(g) < WINDOW_SIZE:
        continue  # skip very short subjects

    # Per-subject normalization
    mean = feats.mean(axis=0, keepdims=True)
    std  = feats.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    feats = (feats - mean) / std

    # Sliding windows within this subject
    for i in range(len(g) - WINDOW_SIZE + 1):
        x_win = feats[i:i + WINDOW_SIZE]            # (T, F)
        y_win = labels[i + WINDOW_SIZE - 1]         # label at last step
        day   = days[i + WINDOW_SIZE - 1]           # day of last step
        sub   = subjects[i + WINDOW_SIZE - 1]       # subject (sid)

        X_list_B.append(x_win)
        y_list_B.append(y_win)
        event_list_B.append(day)
        subject_list_B.append(sub)

# Convert to arrays
X_B           = np.stack(X_list_B, axis=0)          # (N_B, 32, 3)
y_B           = np.array(y_list_B)                  # (N_B,)
event_ids_B   = np.array(event_list_B)              # (N_B,) days
subject_ids_B = np.array(subject_list_B)            # (N_B,) subjects

# Safety: clean NaNs/infs
X_B = np.nan_to_num(X_B, nan=0.0, posinf=0.0, neginf=0.0)

print("Windows shape:", X_B.shape)
print("Labels shape:", y_B.shape)
print("Class balance:", np.bincount(y_B))

unique_days, counts_days = np.unique(event_ids_B, return_counts=True)
print("Unique days and window counts:")
for d, c in zip(unique_days, counts_days):
    print(d, "->", c, "windows")

unique_subj, counts_subj = np.unique(subject_ids_B, return_counts=True)
print("\nUnique subjects and window counts:")
for s, c in zip(unique_subj, counts_subj):
    print(s, "->", c, "windows")


from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch

class WindowDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)   # (N, T, F)
        self.y = torch.from_numpy(y)   # (N,)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

Xb_train, Xb_val, yb_train, yb_val = train_test_split(
    X_B, y_B, test_size=0.2, random_state=42, stratify=y_B
)

trainB_ds = WindowDataset(Xb_train, yb_train)
valB_ds   = WindowDataset(Xb_val,   yb_val)

trainB_loader = DataLoader(trainB_ds, batch_size=64, shuffle=True)
valB_loader   = DataLoader(valB_ds,   batch_size=64, shuffle=False)


import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.BCEWithLogitsLoss()

def run_epoch(model, loader, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss, total, correct = 0.0, 0, 0

    with torch.set_grad_enabled(training):
        for xb, yb in loader:
            xb = xb.to(device)                 # (B, T, F)
            yb = yb.float().to(device)         # (B,)

            if training:
                optimizer.zero_grad()

            logits = model(xb)                 # (B,)
            loss = criterion(logits, yb)

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * xb.size(0)
            total += xb.size(0)
            preds = (torch.sigmoid(logits) > 0.5).long()
            correct += (preds.cpu() == yb.long().cpu()).sum().item()

    return total_loss / total, correct / total

# after training on taxi:
# Same CNNTravelState definition as before
student_model_frozen = CNNTravelState(in_channels=3, hidden_channels=32).to(device)
state_dict = torch.load("cnn_taxi_pretrained.pt", map_location=device)
student_model_frozen.load_state_dict(state_dict)

# Freeze conv layers
for p in student_model_frozen.conv.parameters():
    p.requires_grad = False

optimizer_frozen = optim.Adam(
    filter(lambda p: p.requires_grad, student_model_frozen.parameters()),
    lr=1e-3
)

for epoch in range(1, 11):
    train_loss, train_acc = run_epoch(student_model_frozen, trainB_loader, optimizer_frozen)
    val_loss, val_acc = run_epoch(student_model_frozen, valB_loader, optimizer=None)
    print(f"[StudentLife FROZEN] Epoch {epoch:02d} | "
          f"train loss {train_loss:.4f}, acc {train_acc:.3f} | "
          f"val loss {val_loss:.4f}, acc {val_acc:.3f}")

student_model_full = CNNTravelState(in_channels=3, hidden_channels=32).to(device)
student_model_full.load_state_dict(torch.load("cnn_taxi_pretrained.pt", map_location=device))

for p in student_model_full.parameters():
    p.requires_grad = True

optimizer_full = optim.Adam(student_model_full.parameters(), lr=1e-4)  # smaller LR

for epoch in range(1, 11):
    train_loss, train_acc = run_epoch(student_model_full, trainB_loader, optimizer_full)
    val_loss, val_acc = run_epoch(student_model_full, valB_loader, optimizer=None)
    print(f"[StudentLife FULL] Epoch {epoch:02d} | "
          f"train loss {train_loss:.4f}, acc {train_acc:.3f} | "
          f"val loss {val_loss:.4f}, acc {val_acc:.3f}")


