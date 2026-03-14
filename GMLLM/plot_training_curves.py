# -*- coding: utf-8 -*-
import re
import matplotlib.pyplot as plt

# 读取log文件
log_path = '../log/train_base_model_0303.log'

epochs = []
losses = []
val_f1s = []

with open(log_path, 'r') as f:
    for line in f:
        # 匹配 Epoch, Loss, ValF1
        match = re.search(r'Epoch (\d+) \| Loss ([\d.]+) \|.*?ValF1 ([\d.]+)', line)
        if match:
            epochs.append(int(match.group(1)))
            losses.append(float(match.group(2)))
            val_f1s.append(float(match.group(3)))

# 绘图
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 图1: Loss
axes[0].plot(epochs, losses, 'b-o', markersize=4)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training Loss over Epochs')
axes[0].grid(True, alpha=0.3)

# 图2: ValF1
axes[1].plot(epochs, val_f1s, 'r-o', markersize=4)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('ValF1')
axes[1].set_title('Validation F1 over Epochs')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../log/training_curves.png', dpi=150)
plt.show()

print(f"Total epochs: {len(epochs)}")
print(f"Final Loss: {losses[-1]:.4f}")
print(f"Final ValF1: {val_f1s[-1]:.4f}")
print(f"Best ValF1: {max(val_f1s):.4f} at epoch {val_f1s.index(max(val_f1s)) + 1}")
