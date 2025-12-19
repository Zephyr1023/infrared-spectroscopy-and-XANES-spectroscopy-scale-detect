import os
import random
import cv2
import glob
import numpy as np
from ultralytics import YOLO

# ================= 配置区域 =================
# 1. 模型路径 (使用你刚训练完的混合模型)
MODEL_PATH = r"/chart_training/mixed_L_Final(third)\weights\best.pt"

# 2. 测试集目录
TEST_DIR = r"D:\图表数字化\数据\Ramen test"

# 3. 结果保存目录
OUTPUT_DIR = "inference_results_third"

# 4. 置信度阈值 (Sim2Real 建议稍微放宽一点点，0.25 或 0.3)
CONF_THRESHOLD = 0.3


# ===========================================

def draw_prediction(img, axis_data, axis_name, color):
    """
    在图上绘制 Top 2 的点和连线
    axis_data: [(conf, kpts), ...]
    color: (B, G, R)
    """
    h, w = img.shape[:2]
    status_text = f"{axis_name}: OK"

    # 如果检测到的有效点少于 2 个，判定为缺失
    if len(axis_data) < 2:
        status_text = f"{axis_name}: Missing/Low Conf"
        print(f"   ⚠️ 警告: {axis_name} 轴检测到的刻度对不足 2 个，可能缺少数值标注。")

    # 只取前两名 (Top 2)
    top2 = axis_data[:2]

    for i, (conf, kpts) in enumerate(top2):
        # kpts shape: [2, 2] -> [[mark_x, mark_y], [text_x, text_y]]
        # 注意：YOLOv11 输出的 kpts 通常包含 x,y,conf (如果 dim=3) 或 x,y (如果 dim=2)
        # 这里假设是 Tensor 转换来的 numpy array

        # 提取关键点 1 (刻度线 Mark)
        mx, my = int(kpts[0][0]), int(kpts[0][1])
        # 提取关键点 2 (数字中心 Text)
        tx, ty = int(kpts[1][0]), int(kpts[1][1])

        # 1. 画点
        cv2.circle(img, (mx, my), 5, (0, 255, 0), -1)  # 绿色: 刻度
        cv2.circle(img, (tx, ty), 5, (0, 255, 255), -1)  # 黄色: 数字

        # 2. 画连线
        cv2.line(img, (mx, my), (tx, ty), color, 2)

        # 3. 标序号 (1st, 2nd)
        label = f"{axis_name}{i + 1} ({conf:.2f})"
        cv2.putText(img, label, (mx, my - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return status_text


def run_inference():
    # 1. 准备环境
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 2. 加载模型
    print(f"正在加载模型: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    # 3. 随机选取图片
    all_imgs = glob.glob(os.path.join(TEST_DIR, "*.jpg")) + \
               glob.glob(os.path.join(TEST_DIR, "*.png"))

    if len(all_imgs) < 10:
        selected_imgs = all_imgs
    else:
        selected_imgs = random.sample(all_imgs, 10)

    print(f"已选中 {len(selected_imgs)} 张图片进行测试...\n")

    # 4. 开始推理
    for img_path in selected_imgs:
        file_name = os.path.basename(img_path)
        print(f"正在处理: {file_name}")

        # 读取原图用于绘制
        original_img = cv2.imread(img_path)
        if original_img is None: continue

        # 推理
        results = model.predict(img_path, conf=CONF_THRESHOLD, verbose=False)[0]

        # 分离 X 轴和 Y 轴的数据
        # 格式: (置信度, 关键点坐标)
        x_axis_candidates = []
        y_axis_candidates = []

        # 解析结果
        if results.boxes is not None and results.keypoints is not None:
            boxes = results.boxes
            kpts = results.keypoints.xy.cpu().numpy()  # 获取坐标

            for i, box in enumerate(boxes):
                cls_id = int(box.cls[0])  # 0: x_axis, 1: y_axis
                conf = float(box.conf[0])
                current_kpts = kpts[i]  # [[x1, y1], [x2, y2]]

                # 简单过滤：如果关键点坐标是 (0,0) 说明没检测到点
                if np.sum(current_kpts) < 1:
                    continue

                if cls_id == 0:
                    x_axis_candidates.append((conf, current_kpts))
                elif cls_id == 1:
                    y_axis_candidates.append((conf, current_kpts))

        # 按置信度排序 (从大到小)
        x_axis_candidates.sort(key=lambda x: x[0], reverse=True)
        y_axis_candidates.sort(key=lambda x: x[0], reverse=True)

        # 绘制并获取状态
        # 蓝色 (255, 0, 0) 表示 X轴
        status_x = draw_prediction(original_img, x_axis_candidates, "X", (255, 0, 0))
        # 红色 (0, 0, 255) 表示 Y轴
        status_y = draw_prediction(original_img, y_axis_candidates, "Y", (0, 0, 255))

        # 在图片左上角打印最终判断
        info_text = f"{status_x} | {status_y}"
        cv2.putText(original_img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 4)  # 黑色描边
        cv2.putText(original_img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)  # 白色文字

        # 保存
        save_path = os.path.join(OUTPUT_DIR, f"result_{file_name}")
        cv2.imwrite(save_path, original_img)
        print(f"   -> 结果已保存: {save_path}")

    print(f"\n✅ 全部完成！请查看文件夹: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    run_inference()