import cv2
import os
import glob
import numpy as np
from tqdm import tqdm

# ================= 配置区域 =================
# 1. 验证集的图片文件夹
IMG_DIR = r"dataset_v8_final/val_real image and json"

# 2. 验证集的标签文件夹 (如果和图片在同一个文件夹，就填一样的路径)
# 假设按照 YOLO 惯例，你的标签可能在同级目录或 labels 文件夹下
# 如果你的 txt 就在 dataset_v8_final/val_real txt txt 里，这里就保持不变
LABEL_DIR = r"dataset_v8_final/val_real txt txt"

# 3. 可视化结果保存路径
OUT_DIR = r"dataset_v8_final/val_real_visualized"

# ===========================================

# 定义颜色 (B, G, R)
COLOR_X_AXIS = (255, 0, 0)  # 蓝色表示 X轴
COLOR_Y_AXIS = (0, 0, 255)  # 红色表示 Y轴
COLOR_MARK = (0, 255, 0)  # 绿色点表示 刻度线 (Tick Mark)
COLOR_TEXT = (0, 255, 255)  # 黄色点表示 文字 (Text Center)


def denormalize(val, max_val):
    return int(float(val) * max_val)


def visualize():
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    # 支持的图片格式
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    img_paths = []
    for ext in extensions:
        img_paths.extend(glob.glob(os.path.join(IMG_DIR, ext)))

    print(f"🔍 找到 {len(img_paths)} 张图片，开始验证...")

    for img_path in tqdm(img_paths):
        # 读取图片
        img = cv2.imread(img_path)
        if img is None:
            continue

        h_img, w_img = img.shape[:2]

        # 寻找对应的 txt 文件
        # 假设文件名一致: image.jpg -> image.txt
        basename = os.path.splitext(os.path.basename(img_path))[0]
        txt_path = os.path.join(LABEL_DIR, basename + ".txt")

        if not os.path.exists(txt_path):
            # 如果在 LABEL_DIR 找不到，尝试去 labels 文件夹找 (常见的 YOLO 结构)
            # 这里的逻辑根据你的实际目录结构调整
            txt_path_alt = txt_path.replace("synthetic images", "labels")
            if os.path.exists(txt_path_alt):
                txt_path = txt_path_alt
            else:
                # 确实没有标签，跳过
                continue

        with open(txt_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5: continue

            # 解析 YOLO 格式
            # <class> <cx> <cy> <w> <h> <px1> <py1> <v1> <px2> <py2> <v2>
            cls_id = int(parts[0])

            # 1. 解析 Bounding Box
            n_cx, n_cy, n_w, n_h = map(float, parts[1:5])

            cx = denormalize(n_cx, w_img)
            cy = denormalize(n_cy, h_img)
            w = denormalize(n_w, w_img)
            h = denormalize(n_h, h_img)

            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            x2 = int(cx + w / 2)
            y2 = int(cy + h / 2)

            # 选择颜色
            color = COLOR_X_AXIS if cls_id == 0 else COLOR_Y_AXIS
            label_name = "X" if cls_id == 0 else "Y"

            # 画框
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label_name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # 2. 解析 Keypoints (如果有)
            # 你的格式应该是 11 个数 (class + box + kpt1 + kpt2)
            if len(parts) >= 11:
                # Kpt 1: Tick Mark (刻度线)
                kx1 = denormalize(parts[5], w_img)
                ky1 = denormalize(parts[6], h_img)

                # Kpt 2: Text Center (文字)
                kx2 = denormalize(parts[8], w_img)
                ky2 = denormalize(parts[9], h_img)

                # 画点
                # 绿色实心圆 = 刻度线
                cv2.circle(img, (kx1, ky1), 4, COLOR_MARK, -1)
                # 黄色实心圆 = 文字中心
                cv2.circle(img, (kx2, ky2), 4, COLOR_TEXT, -1)

                # 画一条线连接它们，方便看配对是否正确
                cv2.line(img, (kx1, ky1), (kx2, ky2), (200, 200, 200), 1, cv2.LINE_AA)

        # 保存图片
        out_path = os.path.join(OUT_DIR, os.path.basename(img_path))
        cv2.imwrite(out_path, img)

    print(f"✅ 验证完成！请打开文件夹检查: {OUT_DIR}")
    print("图例说明:")
    print("🟦 蓝色框: X轴数据")
    print("🟥 红色框: Y轴数据")
    print("🟢 绿色点: 刻度线 (Tick Mark)")
    print("🟡 黄色点: 数字中心 (Tick Label)")


if __name__ == "__main__":
    visualize()