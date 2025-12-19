import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os
import random

# --- 配置路径 ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 对应 dataset_v8_final 的目录结构
IMG_DIR = os.path.join(CURRENT_DIR, "dataset_v8_final/synthetic images")
POSE_TXT_DIR = os.path.join(CURRENT_DIR, "dataset_v8_final/synthetic labels_pose")

print(f"Reading Images from: {IMG_DIR}")
print(f"Reading Pose Labels from: {POSE_TXT_DIR}")


def visualize_pose(filename):
    img_path = os.path.join(IMG_DIR, filename)
    txt_filename = os.path.splitext(filename)[0] + ".txt"
    txt_path = os.path.join(POSE_TXT_DIR, txt_filename)

    if not os.path.exists(img_path):
        print(f"Error: Image not found -> {filename}")
        return
    if not os.path.exists(txt_path):
        print(f"Error: Pose Label not found -> {txt_filename} (请确保运行了生成 Pose 的代码)")
        return

    try:
        image = Image.open(img_path)
        img_w, img_h = image.size

        with open(txt_path, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"File load error: {e}")
        return

    # 创建画布
    fig, ax = plt.subplots(figsize=(14, 8))  # 稍微加宽一点画布，方便放图例
    ax.imshow(image)
    ax.set_title(f"Check Pose: {filename} ({len(lines)} pairs)", fontsize=12)

    # --- 解析 YOLO Pose 格式 ---
    for line in lines:
        parts = list(map(float, line.strip().split()))
        if len(parts) < 11: continue

        cls_id = int(parts[0])

        # 1. BBox (外框)
        box_xc, box_yc, box_w, box_h = parts[1:5]
        x1 = (box_xc - box_w / 2) * img_w
        y1 = (box_yc - box_h / 2) * img_h
        w = box_w * img_w
        h = box_h * img_h

        box_color = 'yellow' if cls_id == 0 else 'cyan'
        rect = patches.Rectangle((x1, y1), w, h, linewidth=1, edgecolor=box_color,
                                 facecolor='none', linestyle='--', alpha=0.6)
        ax.add_patch(rect)

        # 2. Keypoints (关键点)
        k1_x, k1_y, k1_v = parts[5], parts[6], parts[7]
        px1, py1 = k1_x * img_w, k1_y * img_h

        k2_x, k2_y, k2_v = parts[8], parts[9], parts[10]
        px2, py2 = k2_x * img_w, k2_y * img_h

        # 3. 绘制
        if k1_v > 0 and k2_v > 0:
            ax.plot([px1, px2], [py1, py2], color=box_color, linewidth=1.5, alpha=0.8)
            ax.scatter(px1, py1, c='#00FF00', s=20, zorder=10, edgecolors='black', linewidth=0.5, marker='o')
            ax.scatter(px2, py2, c='#FF0000', s=20, zorder=10, edgecolors='black', linewidth=0.5, marker='s')

    # --- [修改部分] 图例说明 ---
    legend_text = (
        "Visual Guide:\n"
        "----------------\n"
        "Yellow: X-Axis Pair\n"
        "Cyan:   Y-Axis Pair\n\n"
        "Keypoints:\n"
        "● Green Dot:  Tick Mark\n"
        "■ Red Square: Tick Label\n\n"
        "Line: Connects Mark to Label"
    )

    # 将位置设置为 (1.02, 1.0)，即 Axes 的右侧外部
    ax.text(1.02, 1.0, legend_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f0f0', alpha=1.0, edgecolor='gray'))

    ax.axis('off')

    # 使用 rect 参数为右侧的文字预留 20% 的空间 (0 ~ 0.8 用于画图)
    plt.tight_layout(rect=[0, 0, 0.8, 1])
    plt.show()


# --- 主程序 ---
if __name__ == "__main__":
    if not os.path.exists(IMG_DIR):
        print(f"Error: Directory {IMG_DIR} does not exist!")
    else:
        files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith('.jpg')]

        if not files:
            print("No synthetic images found.")
        else:
            print(f"Found {len(files)} synthetic images. Showing 5 random samples...")
            random.shuffle(files)
            samples = files[:5]

            for fname in samples:
                visualize_pose(fname)