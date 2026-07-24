"""
Model comparison script
Trains all three architectures (2-conv / 4-conv / 6-conv) with identical
hyperparameters and writes performance metrics to CSV files.

Output:
    output/compare_epoch.csv    per-epoch train/val loss and accuracy
    output/compare_summary.csv  best/final metrics and parameter counts
"""

import os
import time
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

EPOCHS     = 100
BATCH_SIZE = 64
LR         = 0.001
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {DEVICE}")


# ------------------------------------------------------------------ #
# Model definitions
# ------------------------------------------------------------------ #
class CNN2Conv(nn.Module):
    """Conv(3->16) -> Pool -> Conv(16->32) -> Pool -> FC"""
    def __init__(self):
        super().__init__()
        self.relu  = nn.ReLU()
        self.conv1 = nn.Conv2d(3,  16, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(32 * 8 * 8, 256)
        self.fc2   = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


class CNN4Conv(nn.Module):
    """2-conv + Conv(32->64) -> Pool -> Conv(64->128) -> Pool -> FC"""
    def __init__(self):
        super().__init__()
        self.relu  = nn.ReLU()
        self.conv1 = nn.Conv2d(3,   16,  3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16,  32,  3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32,  64,  3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(64,  128, 3, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(128 * 2 * 2, 256)
        self.fc2   = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.pool3(self.relu(self.conv3(x)))
        x = self.pool4(self.relu(self.conv4(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


class CNN6Conv(nn.Module):
    """4-conv + Conv(128->256, no pool) + Conv(256->256, no pool) -> FC"""
    def __init__(self):
        super().__init__()
        self.relu  = nn.ReLU()
        self.conv1 = nn.Conv2d(3,   16,  3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16,  32,  3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32,  64,  3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(64,  128, 3, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.fc1   = nn.Linear(256 * 2 * 2, 256)
        self.fc2   = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.pool3(self.relu(self.conv3(x)))
        x = self.pool4(self.relu(self.conv4(x)))
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


MODELS = [
    ("2conv",  CNN2Conv),
    ("4conv",  CNN4Conv),
    ("6conv",  CNN6Conv),
]


# ------------------------------------------------------------------ #
# Data
# ------------------------------------------------------------------ #
def get_dataloaders():
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    data_root = os.path.join(os.path.dirname(__file__), "data")
    trainset = torchvision.datasets.CIFAR10(
        root=data_root, train=True,  download=True, transform=transform_train)
    testset  = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=transform_test)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    test_loader  = torch.utils.data.DataLoader(
        testset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return train_loader, test_loader


# ------------------------------------------------------------------ #
# Training
# ------------------------------------------------------------------ #
def train_one_model(model, train_loader, test_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            _, predicted  = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total   += images.size(0)

        train_loss = running_loss / total
        train_acc  = correct / total

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss    += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total   += images.size(0)

        val_loss /= val_total
        val_acc   = val_correct / val_total
        history.append((epoch, train_loss, train_acc, val_loss, val_acc))

        print(f"  Epoch [{epoch:2d}/{EPOCHS}]  "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}  |  "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}")

    return history


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    train_loader, test_loader = get_dataloaders()

    epoch_rows   = []
    summary_rows = []

    for model_name, ModelClass in MODELS:
        print(f"\n{'='*60}")
        print(f" Training {model_name}")
        print(f"{'='*60}")

        model  = ModelClass().to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")

        t0      = time.time()
        history = train_one_model(model, train_loader, test_loader)
        elapsed = time.time() - t0

        for (epoch, tl, ta, vl, va) in history:
            epoch_rows.append({
                "model":      model_name,
                "epoch":      epoch,
                "train_loss": round(tl, 6),
                "train_acc":  round(ta, 6),
                "val_loss":   round(vl, 6),
                "val_acc":    round(va, 6),
            })

        best_epoch, _, _, _, best_val_acc = max(history, key=lambda r: r[4])
        final = history[-1]
        summary_rows.append({
            "model":          model_name,
            "n_params":       n_params,
            "best_val_acc":   round(best_val_acc, 6),
            "best_epoch":     best_epoch,
            "final_val_acc":  round(final[4], 6),
            "final_val_loss": round(final[3], 6),
            "train_time_sec": round(elapsed, 1),
        })

    # write epoch CSV
    epoch_path = os.path.join(OUTPUT_DIR, "compare_epoch.csv")
    with open(epoch_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=epoch_rows[0].keys())
        writer.writeheader()
        writer.writerows(epoch_rows)
    print(f"\nSaved: {epoch_path}")

    # write summary CSV
    summary_path = os.path.join(OUTPUT_DIR, "compare_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Saved: {summary_path}")

    # print summary table to console
    print(f"\n{'─'*75}")
    print(f"{'model':<8} {'params':>10} {'best_val_acc':>13} {'best_epoch':>11} "
          f"{'final_val_acc':>14} {'time(s)':>9}")
    print(f"{'─'*75}")
    for r in summary_rows:
        print(f"{r['model']:<8} {r['n_params']:>10,} {r['best_val_acc']:>13.4f} "
              f"{r['best_epoch']:>11} {r['final_val_acc']:>14.4f} "
              f"{r['train_time_sec']:>9.1f}")
    print(f"{'─'*75}")
