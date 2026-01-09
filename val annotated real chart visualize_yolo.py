import cv2
import os
import glob
import numpy as np
from tqdm import tqdm

# ================= 配置区域 =================
# 1. 验证集的图片文件夹
IMG_DIR = r"/Fourth Raw Dataset"

# 2. 验证集的标签文件夹
LABEL_DIR = r"/Fourth Raw Dataset_txt"

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


# --- 新增：读取中文路径图片的辅助函数 ---
def cv2_imread_chinese(file_path):
    try:
        # np.fromfile 读取文件流，cv2.imdecode 解码
        img_array = np.fromfile(file_path, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


# --- 新增：保存中文路径图片的辅助函数 ---
def cv2_imwrite_chinese(file_path, img):
    try:
        # 获取文件后缀名 (如 .jpg)
        ext = os.path.splitext(file_path)[1]
        if not ext:
            ext = ".jpg"  # 默认后缀
        # cv2.imencode 编码，tofile 保存
        cv2.imencode(ext, img)[1].tofile(file_path)
    except Exception as e:
        print(f"Error writing {file_path}: {e}")


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
        # 1. 修改处：使用自定义函数读取带中文路径的图片
        img = cv2_imread_chinese(img_path)

        if img is None:
            print(f"⚠️ 无法读取图片: {img_path}")
            continue

        h_img, w_img = img.shape[:2]

        # 寻找对应的 txt 文件
        basename = os.path.splitext(os.path.basename(img_path))[0]
        txt_path = os.path.join(LABEL_DIR, basename + ".txt")

        # 简单的文件存在检查
        if not os.path.exists(txt_path):
            # 如果你有备用路径逻辑可以加在这里，暂时直接跳过
            # print(f"未找到标签: {txt_path}")
            continue

        with open(txt_path, 'r', encoding='utf-8') as f:  # 建议加上 encoding='utf-8' 防止读txt报错
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5: continue

            # 解析 YOLO 格式
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

            # 2. 解析 Keypoints
            if len(parts) >= 11:
                # Kpt 1: Tick Mark
                kx1 = denormalize(parts[5], w_img)
                ky1 = denormalize(parts[6], h_img)

                # Kpt 2: Text Center
                kx2 = denormalize(parts[8], w_img)
                ky2 = denormalize(parts[9], h_img)

                # 画点
                cv2.circle(img, (kx1, ky1), 4, COLOR_MARK, -1)
                cv2.circle(img, (kx2, ky2), 4, COLOR_TEXT, -1)

                # 画线
                cv2.line(img, (kx1, ky1), (kx2, ky2), (200, 200, 200), 1, cv2.LINE_AA)

        # 2. 修改处：使用自定义函数保存带中文路径的图片
        out_path = os.path.join(OUT_DIR, os.path.basename(img_path))
        cv2_imwrite_chinese(out_path, img)

    print(f"✅ 验证完成！请打开文件夹检查: {OUT_DIR}")


if __name__ == "__main__":
    visualize()