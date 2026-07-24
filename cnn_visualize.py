"""
CNN Training & Visualization Sample (study session)
Dataset : CIFAR-10 (auto-download)
Network :
    Conv1(3->16,  3x3) -> ReLU -> MaxPool(2x2)   32x32 -> 16x16
    Conv2(16->32, 3x3) -> ReLU -> MaxPool(2x2)   16x16 ->  8x8
    Conv3(32->64, 3x3) -> ReLU -> MaxPool(2x2)    8x8  ->  4x4
    Conv4(64->128,3x3) -> ReLU -> MaxPool(2x2)    4x4  ->  2x2
    Flatten -> FC(128x2x2->256) -> ReLU -> FC(256->10)

Output files:
    output/01_network_architecture.png  network architecture diagram
    output/02_conv1_filters.png         Conv1 filter weight visualization
    output/03_training_curve.png        training curve (Loss / Accuracy)
    output/04_conv1_feature_maps.png    Conv1 feature maps (single test image)
    output/05_conv2_feature_maps.png    Conv2 feature maps (single test image)
    output/06_conv3_feature_maps.png    Conv3 feature maps (single test image)
    output/07_conv4_feature_maps.png    Conv4 feature maps (single test image)
    output/08_activation_comparison.png activation distribution before/after ReLU
    output/09_test_predictions.png      test image prediction results
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use("Agg")  # non-interactive backend (no GUI required)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

#  output directory 
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

#  hyperparameters 
EPOCHS      = 10
BATCH_SIZE  = 64
LR          = 0.001
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {DEVICE}")


# PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP
# 1. Network definition
# PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP
class SimpleCNN(nn.Module):
    """
    Conv(3->16)   -> ReLU -> MaxPool   32x32 -> 16x16
    Conv(16->32)  -> ReLU -> MaxPool   16x16 ->  8x8
    Conv(32->64)  -> ReLU -> MaxPool    8x8  ->  4x4
    Conv(64->128) -> ReLU -> MaxPool    4x4  ->  2x2
    FC(128x2x2->256) -> ReLU -> FC(256->10)
    """
    def __init__(self):
        super().__init__()
        # block 1
        self.conv1 = nn.Conv2d(in_channels=3,   out_channels=16,
                               kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # block 2
        self.conv2 = nn.Conv2d(in_channels=16,  out_channels=32,
                               kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # block 3
        self.conv3 = nn.Conv2d(in_channels=32,  out_channels=64,
                               kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # block 4
        self.conv4 = nn.Conv2d(in_channels=64,  out_channels=128,
                               kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # fully connected layers
        # CIFAR-10: 32x32 -> 16x16 -> 8x8 -> 4x4 -> 2x2
        self.fc1 = nn.Linear(128 * 2 * 2, 256)
        self.fc2 = nn.Linear(256, 10)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.pool3(self.relu(self.conv3(x)))
        x = self.pool4(self.relu(self.conv4(x)))
        x = x.view(x.size(0), -1)   # Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    #  returns intermediate outputs as a dict (for visualization)
    def get_feature_maps(self, x):
        """Return per-layer outputs as a dictionary."""
        out = {}
        after_conv1 = self.conv1(x)
        after_relu1 = self.relu(after_conv1)
        after_pool1 = self.pool1(after_relu1)

        after_conv2 = self.conv2(after_pool1)
        after_relu2 = self.relu(after_conv2)
        after_pool2 = self.pool2(after_relu2)

        after_conv3 = self.conv3(after_pool2)
        after_relu3 = self.relu(after_conv3)
        after_pool3 = self.pool3(after_relu3)

        after_conv4 = self.conv4(after_pool3)
        after_relu4 = self.relu(after_conv4)
        after_pool4 = self.pool4(after_relu4)

        out["conv1_before_relu"] = after_conv1.detach()
        out["conv1_after_relu"]  = after_relu1.detach()
        out["pool1"]             = after_pool1.detach()
        out["conv2_before_relu"] = after_conv2.detach()
        out["conv2_after_relu"]  = after_relu2.detach()
        out["pool2"]             = after_pool2.detach()
        out["conv3_before_relu"] = after_conv3.detach()
        out["conv3_after_relu"]  = after_relu3.detach()
        out["pool3"]             = after_pool3.detach()
        out["conv4_before_relu"] = after_conv4.detach()
        out["conv4_after_relu"]  = after_relu4.detach()
        out["pool4"]             = after_pool4.detach()
        return out


# PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP
# 2. Data preparation (CIFAR-10)
# PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP
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
        root=data_root, train=True, download=True, transform=transform_train)
    testset  = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader  = torch.utils.data.DataLoader(
        testset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return train_loader, test_loader, testset


# PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP
# 3. Training
# PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP
def train(model, train_loader, test_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, EPOCHS + 1):
        #  training phase 
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
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total   += images.size(0)

        train_loss = running_loss / total
        train_acc  = correct / total

        #  validation phase 
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

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch [{epoch:2d}/{EPOCHS}]  "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}  |  "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}")

    return history


# PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP
# 4. Visualization functions
# PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP

def save_network_architecture():
    """Save a diagram of the network architecture."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis("off")
    fig.patch.set_facecolor("#F8F9FA")

    # block definition: (x_center, y_center, width, height, label, color)
    blocks = [
        (0.04, 0.5, 0.05, 0.35, "Input\n32x32x3",       "#AED6F1"),
        (0.13, 0.5, 0.05, 0.35, "Conv1\n3->16\n3x3",    "#A9DFBF"),
        (0.21, 0.5, 0.04, 0.28, "ReLU",                 "#F9E79F"),
        (0.29, 0.5, 0.05, 0.28, "MaxPool\n16x16x16",    "#FAD7A0"),
        (0.38, 0.5, 0.05, 0.25, "Conv2\n16->32\n3x3",   "#A9DFBF"),
        (0.46, 0.5, 0.04, 0.20, "ReLU",                 "#F9E79F"),
        (0.54, 0.5, 0.05, 0.20, "MaxPool\n8x8x32",      "#FAD7A0"),
        (0.63, 0.5, 0.05, 0.18, "Conv3\n32->64\n3x3",   "#A9DFBF"),
        (0.71, 0.5, 0.04, 0.15, "ReLU",                 "#F9E79F"),
        (0.79, 0.5, 0.05, 0.15, "MaxPool\n4x4x64",      "#FAD7A0"),
        (0.87, 0.5, 0.05, 0.13, "Conv4\n64->128\n3x3",  "#A9DFBF"),
        (0.93, 0.5, 0.04, 0.12, "ReLU+Pool\n2x2x128",  "#FAD7A0"),
        (0.99, 0.5, 0.05, 0.35, "FC\n->256->10",        "#D2B4DE"),
    ]

    for (x, y, w, h, label, color) in blocks:
        rect = plt.Rectangle((x - w/2, y - h/2), w, h,
                              linewidth=1.5, edgecolor="#555",
                              facecolor=color, zorder=3)
        ax.add_patch(rect)
        ax.text(x, y, label, ha="center", va="center",
                fontsize=8, fontweight="bold", zorder=4)

    # arrows between blocks
    for i in range(len(blocks) - 1):
        x1 = blocks[i][0]   + blocks[i][2] / 2
        x2 = blocks[i+1][0] - blocks[i+1][2] / 2
        y  = 0.5
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle="->", color="#333", lw=1.5))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("SimpleCNN Network Architecture (4 Conv Blocks)", fontsize=13, fontweight="bold", pad=10)

    path = os.path.join(OUTPUT_DIR, "01_network_architecture.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def save_conv1_filters(model):
    """Visualize the 16 filter weights of Conv1."""
    weights = model.conv1.weight.data.cpu()  # shape: (16, 3, 3, 3)

    fig, axes = plt.subplots(2, 8, figsize=(14, 4))
    fig.suptitle("Conv1 Filter Weights (16 filters x RGB)\n"
                 "Each cell = one 3x3 filter displayed as RGB",
                 fontsize=11, fontweight="bold")

    for i, ax in enumerate(axes.flat):
        # normalize RGB channels to [0, 1] for display
        filt = weights[i]  # shape: (3, 3, 3)
        filt_np = filt.numpy().transpose(1, 2, 0)  # -> (H, W, C)
        filt_norm = (filt_np - filt_np.min()) / (filt_np.max() - filt_np.min() + 1e-8)
        ax.imshow(filt_norm)
        ax.set_title(f"filter {i}", fontsize=7)
        ax.axis("off")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "02_conv1_filters.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def save_training_curve(history):
    """Save training curves (Loss / Accuracy)."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Training Curve", fontsize=13, fontweight="bold")

    # Loss
    ax1.plot(epochs, history["train_loss"], "o-", label="Train Loss", color="#3498DB")
    ax1.plot(epochs, history["val_loss"],   "s--", label="Val Loss",   color="#E74C3C")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Loss over Epochs")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(epochs, history["train_acc"], "o-", label="Train Acc", color="#2ECC71")
    ax2.plot(epochs, history["val_acc"],   "s--", label="Val Acc",   color="#E67E22")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy over Epochs")
    ax2.set_ylim(0, 1); ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "03_training_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def _denormalize(tensor):
    """Reverse CIFAR-10 normalization for display."""
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std  = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)


def save_feature_maps(model, image_tensor, label_idx, layer="conv1"):
    """Visualize feature maps for the specified layer."""
    model.eval()
    with torch.no_grad():
        feat_maps = model.get_feature_maps(image_tensor.unsqueeze(0).to(DEVICE))

    key_relu = f"{layer}_after_relu"
    fmaps = feat_maps[key_relu][0].cpu()  # (C, H, W)
    n_channels = fmaps.shape[0]           # 16 or 32

    cols = 8
    rows = (n_channels + cols - 1) // cols

    fig = plt.figure(figsize=(14, rows * 1.8 + 2))
    gs  = gridspec.GridSpec(rows + 1, cols + 1,
                            height_ratios=[2] + [1] * rows,
                            width_ratios=[2] + [1] * cols)

    # input image
    ax_img = fig.add_subplot(gs[0, :cols//2])
    orig = _denormalize(image_tensor).numpy().transpose(1, 2, 0)
    ax_img.imshow(orig)
    ax_img.set_title(f"Input Image\nLabel: {CIFAR10_CLASSES[label_idx]}",
                     fontsize=10, fontweight="bold")
    ax_img.axis("off")

    layer_labels = {
        "conv1": "Conv1 (16ch, 16x16)",
        "conv2": "Conv2 (32ch,  8x8)",
        "conv3": "Conv3 (64ch,  4x4)",
        "conv4": "Conv4 (128ch, 2x2)",
    }
    layer_suffixes = {"conv1": "04", "conv2": "05", "conv3": "06", "conv4": "07"}
    fig.suptitle(f"{layer_labels[layer]} Feature Maps (after ReLU)\n"
                 "Brighter = stronger activation",
                 fontsize=11, fontweight="bold", y=1.01)

    for i in range(n_channels):
        r = i // cols
        c = i  % cols
        ax = fig.add_subplot(gs[r + 1, c])
        fmap = fmaps[i].numpy()
        ax.imshow(fmap, cmap="viridis", interpolation="nearest")
        ax.set_title(f"ch{i}", fontsize=6)
        ax.axis("off")

    plt.tight_layout()
    suffix = layer_suffixes[layer]
    path = os.path.join(OUTPUT_DIR, f"{suffix}_{layer}_feature_maps.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def save_activation_comparison(model, image_tensor):
    """Compare activation value distributions before and after ReLU."""
    model.eval()
    with torch.no_grad():
        feat_maps = model.get_feature_maps(image_tensor.unsqueeze(0).to(DEVICE))

    fig, axes = plt.subplots(4, 2, figsize=(12, 16))
    fig.suptitle("Activation Distribution: Before vs After ReLU", fontsize=12, fontweight="bold")

    pairs = [
        ("conv1_before_relu", "conv1_after_relu", "Conv1"),
        ("conv2_before_relu", "conv2_after_relu", "Conv2"),
        ("conv3_before_relu", "conv3_after_relu", "Conv3"),
        ("conv4_before_relu", "conv4_after_relu", "Conv4"),
    ]

    for row, (key_before, key_after, title) in enumerate(pairs):
        before = feat_maps[key_before][0].cpu().numpy().flatten()
        after  = feat_maps[key_after ][0].cpu().numpy().flatten()

        ax_b = axes[row][0]
        ax_a = axes[row][1]

        ax_b.hist(before, bins=60, color="#3498DB", alpha=0.7, edgecolor="white")
        ax_b.set_title(f"{title} Before ReLU", fontsize=10)
        ax_b.set_xlabel("Activation value"); ax_b.set_ylabel("Frequency")
        ax_b.axvline(0, color="red", linestyle="--", linewidth=1.2, label="x=0")
        ax_b.legend()

        ax_a.hist(after, bins=60, color="#2ECC71", alpha=0.7, edgecolor="white")
        ax_a.set_title(f"{title}  After ReLU (negatives clipped to 0)", fontsize=10)
        ax_a.set_xlabel("Activation value"); ax_a.set_ylabel("Frequency")

        zero_ratio = (after == 0).mean() * 100
        ax_a.text(0.65, 0.85, f"Zero ratio: {zero_ratio:.1f}%",
                  transform=ax_a.transAxes, fontsize=9,
                  bbox=dict(boxstyle="round", facecolor="#F9E79F", alpha=0.8))

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "08_activation_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def save_feature_maps_overlay(model, image_tensor, label_idx, layer="conv1"):
    """Overlay each channel's feature map as a heatmap on top of the original image."""
    model.eval()
    with torch.no_grad():
        feat_maps = model.get_feature_maps(image_tensor.unsqueeze(0).to(DEVICE))

    key_relu = f"{layer}_after_relu"
    fmaps = feat_maps[key_relu][0].cpu().numpy()  # (C, H, W)
    n_channels = fmaps.shape[0]
    orig = _denormalize(image_tensor).numpy().transpose(1, 2, 0)  # (32, 32, 3)
    orig_h, orig_w = orig.shape[:2]

    cols = 8
    rows = (n_channels + cols - 1) // cols

    layer_labels = {
        "conv1": "Conv1 (16ch, 16x16)",
        "conv2": "Conv2 (32ch,  8x8)",
        "conv3": "Conv3 (64ch,  4x4)",
        "conv4": "Conv4 (128ch, 2x2)",
    }
    layer_suffixes = {"conv1": "10", "conv2": "11", "conv3": "12", "conv4": "13"}

    fig, axes = plt.subplots(rows + 1, cols,
                             figsize=(cols * 1.8 + 0.6, (rows + 1) * 1.8 + 1))
    fig.suptitle(f"{layer_labels[layer]} — Feature Map Overlay\n"
                 "Heatmap (viridis) overlaid on original image  |  "
                 f"Label: {CIFAR10_CLASSES[label_idx]}",
                 fontsize=11, fontweight="bold")

    # top row: original image repeated for reference
    for c in range(cols):
        axes[0, c].imshow(orig)
        axes[0, c].axis("off")
        if c == 0:
            axes[0, c].set_title("original", fontsize=7, fontweight="bold")

    im_ref = None
    for i in range(n_channels):
        r = i // cols + 1
        c = i  % cols
        ax = axes[r, c]

        fmap = fmaps[i]
        fmap_up = np.repeat(np.repeat(fmap, orig_h // fmap.shape[0] or 1, axis=0),
                            orig_w // fmap.shape[1] or 1, axis=1)
        fmap_up = fmap_up[:orig_h, :orig_w]
        if fmap_up.shape != (orig_h, orig_w):
            pad_h = orig_h - fmap_up.shape[0]
            pad_w = orig_w - fmap_up.shape[1]
            fmap_up = np.pad(fmap_up, ((0, pad_h), (0, pad_w)), mode="edge")

        fmap_norm = (fmap_up - fmap_up.min()) / (fmap_up.max() - fmap_up.min() + 1e-8)

        ax.imshow(orig)
        im_ref = ax.imshow(fmap_norm, cmap="viridis", alpha=0.55,
                           interpolation="bilinear", vmin=0, vmax=1)
        ax.set_title(f"ch{i}", fontsize=6)
        ax.axis("off")

    # hide unused axes in the last row
    for j in range(n_channels % cols or cols, cols):
        axes[-1, j].axis("off")

    plt.tight_layout(rect=[0, 0, 0.92, 1])

    # colorbar: derive position from actual axes layout after tight_layout
    if im_ref is not None:
        pos_top = axes[1, 0].get_position()
        pos_bot = axes[-1, 0].get_position()
        cbar_ax = fig.add_axes([0.93, pos_bot.y0, 0.015,
                                pos_top.y1 - pos_bot.y0])
        cbar = fig.colorbar(im_ref, cax=cbar_ax)
        cbar.set_ticks([0, 0.5, 1])
        cbar.set_ticklabels(["low", "mid", "high"], fontsize=7)
        cbar.set_label("activation", fontsize=7, labelpad=4)
    suffix = layer_suffixes[layer]
    path = os.path.join(OUTPUT_DIR, f"{suffix}_{layer}_overlay.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def save_test_predictions(model, testset, n=16):
    """Visualize predictions on test images."""
    model.eval()
    indices = np.arange(n)

    cols = 8
    rows = n // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 2))
    fig.suptitle("Test Image Predictions\nGreen border = correct / Red border = incorrect",
                 fontsize=11, fontweight="bold")

    for ax, idx in zip(axes.flat, indices):
        image, label = testset[idx]
        with torch.no_grad():
            output = model(image.unsqueeze(0).to(DEVICE))
            pred   = output.argmax(1).item()

        orig = _denormalize(image).numpy().transpose(1, 2, 0)
        ax.imshow(orig)

        color = "#2ECC71" if pred == label else "#E74C3C"
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)

        ax.set_title(f"GT:{CIFAR10_CLASSES[label]}\nPred:{CIFAR10_CLASSES[pred]}",
                     fontsize=6.5,
                     color="black" if pred == label else "#C0392B")
        ax.axis("off")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "09_test_predictions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP
# 5. Main
# PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP
if __name__ == "__main__":
    print("=" * 60)
    print(" CNN Training & Visualization Sample")
    print("=" * 60)

    # step 1: save network architecture diagram (before training)
    print("\n[1/7] Saving network architecture diagram...")
    save_network_architecture()

    # step 2: prepare data
    print("\n[2/7] Preparing CIFAR-10 dataset...")
    train_loader, test_loader, testset = get_dataloaders()
    print(f"  Train: {len(train_loader.dataset)}  Test: {len(test_loader.dataset)}")

    # step 3: instantiate model
    model = SimpleCNN().to(DEVICE)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")

    # step 4: save Conv1 filters before training
    print("\n[3/7] Saving Conv1 filters (before training)...")
    save_conv1_filters(model)

    # step 5: train
    print(f"\n[4/7] Starting training ({EPOCHS} epochs)...")
    history = train(model, train_loader, test_loader)

    # step 6: save training curve
    print("\n[5/7] Saving training curve...")
    save_training_curve(history)

    # step 7: save feature maps and ReLU comparison (single test image)
    print("\n[6/7] Saving feature maps and activation comparison...")
    sample_image, sample_label = testset[0]
    save_feature_maps(model, sample_image, sample_label, layer="conv1")
    save_feature_maps(model, sample_image, sample_label, layer="conv2")
    save_feature_maps(model, sample_image, sample_label, layer="conv3")
    save_feature_maps(model, sample_image, sample_label, layer="conv4")
    save_activation_comparison(model, sample_image)
    save_feature_maps_overlay(model, sample_image, sample_label, layer="conv1")
    save_feature_maps_overlay(model, sample_image, sample_label, layer="conv2")
    save_feature_maps_overlay(model, sample_image, sample_label, layer="conv3")
    save_feature_maps_overlay(model, sample_image, sample_label, layer="conv4")

    # step 8: save test predictions
    print("\n[7/7] Saving test predictions...")
    save_test_predictions(model, testset, n=16)

    print("\n" + "=" * 60)
    print(" Done! Check the output/ folder.")
    print("=" * 60)
    print("\nGenerated files:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print(f"  {f}")