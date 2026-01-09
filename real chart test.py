import os
import cv2
import numpy as np
from ultralytics import YOLO


def run_strict_inference():
    # ---------------- 配置区域 ----------------
    model_path = r"chart_training/mixed_L_Final_v4_DoubleY(fourth)/weights/best.pt"
    if not os.path.exists(model_path):
        print(f"⚠️ Warning: Best weights not found, trying last.pt...")
        model_path = r"chart_training/mixed_L_Final_v4_DoubleY(fourth)/weights/last.pt"

    source_path = r"F:\Spectral Scale Inspection\dataset_v8_final\test4\images"

    # 输出目录
    save_dir = "inference_results_v4_strict"
    os.makedirs(save_dir, exist_ok=True)

    # 可视化与过滤参数
    PAD_SIZE = 200  # 画布扩展大小
    FONT_SCALE = 0.5
    THICKNESS = 1

    # ===【新规则】置信度阈值 ===
    CONF_THRESHOLD = 0.5  # 低于此分数的检测结果将被过滤
    # ----------------------------------------

    print(f"🚀 Loading model from: {model_path}")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    class_names = model.names

    if not os.path.exists(source_path):
        print(f"❌ Source path does not exist.")
        return

    image_files = [f for f in os.listdir(source_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    print(f"📂 Found {len(image_files)} images.")
    print(f"⚖️  Filtering Rules Applied: \n   1. Pair Check (Box + Keypoint)\n   2. Confidence >= {CONF_THRESHOLD}")

    for i, img_file in enumerate(image_files):
        img_path = os.path.join(source_path, img_file)

        if (i + 1) % 10 == 0:
            print(f"Processing {i + 1}/{len(image_files)}: {img_file}")

        # 1. 推理
        # 这里 conf=0.1 保持较低是为了先召回，然后在下面代码中用 CONF_THRESHOLD 严格过滤
        results = model.predict(img_path, imgsz=1024, conf=0.1, verbose=False)
        result = results[0]
        original_img = cv2.imread(img_path)
        if original_img is None: continue

        h, w = original_img.shape[:2]

        # 2. 创建白底画布
        canvas = cv2.copyMakeBorder(original_img, PAD_SIZE, PAD_SIZE, PAD_SIZE, PAD_SIZE,
                                    cv2.BORDER_CONSTANT, value=(255, 255, 255))

        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            cv2.imwrite(os.path.join(save_dir, f"res_{img_file}"), canvas)
            continue

        labels_buffer = {'L': [], 'R': [], 'T': [], 'B': []}

        # --- 第一次遍历：应用双重过滤规则 ---
        for j, box in enumerate(boxes):
            # 获取基础信息
            conf = box.conf[0].item()

            # ===【规则1：置信度过滤】===
            if conf < CONF_THRESHOLD:
                continue

            # 获取 Box 和 Class
            cls_id = int(box.cls[0].item())
            xyxy_orig = box.xyxy[0].cpu().numpy().astype(int)

            # 获取 Keypoint 信息
            kpt_orig = None
            if result.keypoints is not None and len(result.keypoints) > j:
                kp = result.keypoints.xy[j][0].cpu().numpy()
                kx, ky = int(kp[0]), int(kp[1])

                # ===【规则2：成对检测过滤 (Pairing Filter)】===
                # 必须有有效的关键点坐标
                if kx <= 0 or ky <= 0 or kx >= w or ky >= h:
                    continue

                kpt_orig = (kx, ky)
            else:
                continue

            # --- 通过所有过滤，准备绘制 ---

            # 坐标映射
            x1, y1, x2, y2 = xyxy_orig + PAD_SIZE
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            pkx, pky = kpt_orig[0] + PAD_SIZE, kpt_orig[1] + PAD_SIZE

            # 绘制 Box (绿色)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 200, 0), 1)

            # 绘制 Keypoint (红色实心点)
            cv2.circle(canvas, (pkx, pky), 3, (0, 0, 255), -1)

            # 决策标签位置
            dist_l = kpt_orig[0]
            dist_r = w - kpt_orig[0]
            dist_t = kpt_orig[1]
            dist_b = h - kpt_orig[1]
            min_dist = min(dist_l, dist_r, dist_t, dist_b)

            label_text = f"{class_names[cls_id]}:{conf:.2f}"

            item = {
                'box_rect': (x1, y1, x2, y2),
                'kpt_pos': (pkx, pky),
                'text': label_text,
            }

            if min_dist == dist_l:
                item['sort_val'] = pky
                labels_buffer['L'].append(item)
            elif min_dist == dist_r:
                item['sort_val'] = pky
                labels_buffer['R'].append(item)
            elif min_dist == dist_t:
                item['sort_val'] = pkx
                labels_buffer['T'].append(item)
            else:
                item['sort_val'] = pkx
                labels_buffer['B'].append(item)

        # --- 第二次遍历：绘制引线和文字 ---
        def draw_stack(items, edge_code):
            if not items: return
            items.sort(key=lambda x: x['sort_val'])

            last_pos = -1000

            for item in items:
                pkx, pky = item['kpt_pos']
                text = item['text']
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, THICKNESS)

                target_pos = item['sort_val']

                if edge_code in ['L', 'R']:
                    place_y = max(target_pos, last_pos + th + 5)
                    last_pos = place_y

                    if edge_code == 'L':
                        place_x = 20
                        text_org = (place_x, place_y)
                        line_start = (place_x + tw + 2, place_y - th // 2 + 2)
                    else:  # R
                        place_x = canvas.shape[1] - tw - 20
                        text_org = (place_x, place_y)
                        line_start = (place_x - 2, place_y - th // 2 + 2)

                    cv2.putText(canvas, text, text_org, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 0, 0), THICKNESS)
                    cv2.line(canvas, line_start, (pkx, pky), (200, 200, 200), 1)

                else:  # T, B
                    place_x = max(target_pos - tw // 2, last_pos + tw + 15)
                    last_pos = place_x

                    if edge_code == 'T':
                        place_y = PAD_SIZE - 20
                        text_org = (place_x, place_y)
                        line_start = (place_x + tw // 2, place_y + 2)
                    else:  # B
                        place_y = canvas.shape[0] - PAD_SIZE + 20 + th
                        text_org = (place_x, place_y)
                        line_start = (place_x + tw // 2, place_y - th - 2)

                    cv2.putText(canvas, text, text_org, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 0, 0), THICKNESS)
                    cv2.line(canvas, line_start, (pkx, pky), (200, 200, 200), 1)

        draw_stack(labels_buffer['L'], 'L')
        draw_stack(labels_buffer['R'], 'R')
        draw_stack(labels_buffer['T'], 'T')
        draw_stack(labels_buffer['B'], 'B')

        save_path = os.path.join(save_dir, f"res_{img_file}")
        cv2.imwrite(save_path, canvas)

    print(f"\n✅ Done! Filtered results saved in '{save_dir}'.")


if __name__ == '__main__':
    run_strict_inference()