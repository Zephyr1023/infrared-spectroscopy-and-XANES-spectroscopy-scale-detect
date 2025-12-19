import json
import os
import glob
import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm

# ================= 配置区域 =================
# 1. LabelMe JSON 文件夹 (真实标注)
JSON_DIR = r"dataset_v8_final/train_real image and json"

# 2. 输出 TXT 文件夹 (给 YOLO 验证用)
OUT_DIR = r"dataset_v8_final/train_real txt"


# ===========================================

def get_box_center(p1, p2):
    """计算矩形框的中心点"""
    x_min, y_min = min(p1[0], p2[0]), min(p1[1], p2[1])
    x_max, y_max = max(p1[0], p2[0]), max(p1[1], p2[1])
    return (x_min + x_max) / 2, (y_min + y_max) / 2


def convert_mixed():
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    json_files = glob.glob(os.path.join(JSON_DIR, "*.json"))
    print(f"🚀 正在处理 {len(json_files)} 个混合标注文件 (点+框)...")

    for json_file in tqdm(json_files):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        img_w = data['imageWidth']
        img_h = data['imageHeight']

        # 1. 数据分类容器
        # marks 存坐标: [(x, y), ...]
        # texts 存对象: [{'bbox': [minx, miny, maxx, maxy], 'center': (cx, cy)}, ...]
        data_store = {
            'x_mark': [], 'x_text': [],
            'y_mark': [], 'y_text': []
        }

        # 2. 解析 JSON
        for shape in data['shapes']:
            label = shape['label'].lower().strip()
            pts = shape['points']

            # 情况 A: 刻度线 (Point)
            if shape['shape_type'] == 'point':
                if label in ['x_mark', 'y_mark']:
                    data_store[label].append(pts[0])

            # 情况 B: 数字 (Rectangle)
            elif shape['shape_type'] == 'rectangle':
                if label in ['x_text', 'y_text']:
                    # LabelMe 的矩形是两个点 [[x1, y1], [x2, y2]]
                    p1, p2 = pts[0], pts[1]
                    x_min, y_min = min(p1[0], p2[0]), min(p1[1], p2[1])
                    x_max, y_max = max(p1[0], p2[0]), max(p1[1], p2[1])

                    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2

                    data_store[label].append({
                        'bbox': [x_min, y_min, x_max, y_max],
                        'center': [cx, cy]
                    })

        yolo_lines = []

        # 3. 配对逻辑
        def process_axis(marks, text_objs, class_id):
            if not marks or not text_objs:
                return

            marks_arr = np.array(marks)
            # 提取 text 的中心点用于计算距离
            text_centers = np.array([t['center'] for t in text_objs])

            # 计算距离矩阵
            dists = cdist(marks_arr, text_centers)

            # 简单贪婪匹配 (Nearest Neighbor)
            used_texts = set()

            for i in range(len(marks_arr)):
                # 找到离这个 mark 最近的 text
                j = np.argmin(dists[i])
                min_dist = dists[i][j]

                # 阈值：防止匹配到太远的 (比如 1/4 图片宽度)
                if min_dist < (img_w * 0.25):
                    # 这里不做严格的一对一剔除，允许容错，但通常一个mark只配一个text

                    # --- 核心：计算 Union BBox ---
                    mx, my = marks_arr[i]  # 刻度点坐标
                    text_box = text_objs[j]['bbox']  # [x1, y1, x2, y2]
                    tx, ty = text_objs[j]['center']  # 文字中心坐标

                    # 最终的 YOLO 框 = 包含(刻度点) 和 (文字框) 的最小矩形
                    # 你的合成数据逻辑：Box 包含 Mark 和 Text
                    final_x1 = min(mx, text_box[0])
                    final_y1 = min(my, text_box[1])
                    final_x2 = max(mx, text_box[2])
                    final_y2 = max(my, text_box[3])

                    # 稍微加一点 Padding (比如 2 像素)，防止点正好压在边线上
                    pad = 2
                    final_x1 = max(0, final_x1 - pad)
                    final_y1 = max(0, final_y1 - pad)
                    final_x2 = min(img_w, final_x2 + pad)
                    final_y2 = min(img_h, final_y2 + pad)

                    # 转 YOLO 格式 (Center XYWH)
                    box_w = final_x2 - final_x1
                    box_h = final_y2 - final_y1
                    box_cx = final_x1 + box_w / 2
                    box_cy = final_y1 + box_h / 2

                    # 归一化
                    n_cx, n_cy = box_cx / img_w, box_cy / img_h
                    n_w, n_h = box_w / img_w, box_h / img_h

                    # 关键点 1: Mark (x, y)
                    nk1_x, nk1_y = mx / img_w, my / img_h

                    # 关键点 2: Text Center (x, y)
                    nk2_x, nk2_y = tx / img_w, ty / img_h

                    # 写入 line
                    # Class | Box(cx,cy,w,h) | Kpt1(x,y,v) | Kpt2(x,y,v)
                    line = f"{class_id} {n_cx:.6f} {n_cy:.6f} {n_w:.6f} {n_h:.6f} {nk1_x:.6f} {nk1_y:.6f} 2 {nk2_x:.6f} {nk2_y:.6f} 2"
                    yolo_lines.append(line)

        # 执行 X 和 Y 轴
        process_axis(data_store['x_mark'], data_store['x_text'], 0)
        process_axis(data_store['y_mark'], data_store['y_text'], 1)

        # 4. 保存
        txt_name = os.path.basename(json_file).replace('.json', '.txt')
        with open(os.path.join(OUT_DIR, txt_name), 'w') as f_out:
            f_out.write('\n'.join(yolo_lines))

    print(f"✅ 转换完成！输出目录: {OUT_DIR}")


if __name__ == "__main__":
    convert_mixed()