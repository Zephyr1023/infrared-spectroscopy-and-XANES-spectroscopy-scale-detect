import os
import cv2
import torch
import random  # 新增：用于随机抽取
from ultralytics import YOLO


def run_top2_anchors_inference():
    # ---------------- 配置区域 ----------------
    # 1. 模型路径
    model_path = r"chart_training/exp_pose_large_enhanced/weights/best.pt"
    # 如果 best.pt 还没生成，回退到 last.pt
    if not os.path.exists(model_path):
        model_path = r"chart_training/exp_pose_large_enhanced/weights/last.pt"

    # 2. 真实数据路径
    source_path = r"D:\图表数字化\数据\拉曼光谱"

    # 3. 输出目录
    save_dir = "inference_results_second"
    os.makedirs(save_dir, exist_ok=True)

    # 4. 验证数量设置
    sample_count = 20  # 设置需要抽取的图片数量
    # ----------------------------------------

    print(f"🚀 Loading model from: {model_path}")
    model = YOLO(model_path)
    class_names = model.names
    print(f"📋 Class Map: {class_names}")

    # 获取文件夹下所有图片
    all_image_files = [f for f in os.listdir(source_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    total_found = len(all_image_files)

    # ---------------- 随机抽取逻辑 ----------------
    if total_found > sample_count:
        image_files = random.sample(all_image_files, sample_count)
        print(f"📂 Found {total_found} images. Randomly selected {len(image_files)} for validation.")
    else:
        image_files = all_image_files
        print(f"📂 Found {total_found} images (<= {sample_count}). Processing all available images.")
    # --------------------------------------------

    for img_file in image_files:
        img_path = os.path.join(source_path, img_file)

        # 1. 推理
        # conf=0.1: 稍微放宽一点阈值，确保能凑齐至少2个点
        results = model.predict(img_path, imgsz=1024, conf=0.1, verbose=False)
        result = results[0]

        original_img = cv2.imread(img_path)
        if original_img is None: continue

        # 字典用于存储每一类的所有候选者
        # 结构: {class_id: [ {'conf': float, 'box': array, 'kpts': array}, ... ]}
        candidates_pool = {}

        # ---------------- 2. 收集所有检测结果 ----------------
        boxes = result.boxes
        if boxes is not None:
            for i, box in enumerate(boxes):
                cls_id = int(box.cls[0].item())
                conf = box.conf[0].item()
                xyxy = box.xyxy[0].cpu().numpy().astype(int)

                kpts = None
                if result.keypoints is not None and len(result.keypoints) > i:
                    kpts = result.keypoints.xy[i].cpu().numpy()

                if cls_id not in candidates_pool:
                    candidates_pool[cls_id] = []

                candidates_pool[cls_id].append({
                    'conf': conf,
                    'box': xyxy,
                    'kpts': kpts,
                    'name': class_names[cls_id]
                })

        # ---------------- 3. 筛选 Top 2 并可视化 ----------------
        print(f"\nScanning: {img_file}")
        if not candidates_pool:
            print("  ⚠️ No detections found.")
            # 即使没有检测到，也可以保存一下原图以便查看
            # cv2.imwrite(os.path.join(save_dir, f"top2_{img_file}"), original_img)
            continue

        for cls_id, items in candidates_pool.items():
            # 按置信度从高到低排序
            items.sort(key=lambda x: x['conf'], reverse=True)

            # 取前2名 (Top 2)
            top2_items = items[:2]

            name = class_names[cls_id]
            print(f"  🔹 Class: {name} (Found {len(items)}, Keeping Top {len(top2_items)})")

            for rank, data in enumerate(top2_items):
                conf = data['conf']
                box = data['box']
                kpts = data['kpts']

                print(f"     #{rank + 1} Conf: {conf:.4f} | Box: {box}")

                # --- 绘制 Box ---
                # 第一名绿色，第二名黄色，方便区分
                color = (0, 255, 0) if rank == 0 else (0, 255, 255)
                cv2.rectangle(original_img, (box[0], box[1]), (box[2], box[3]), color, 2)

                # --- 绘制 Label 文字 ---
                label_text = f"{name} #{rank + 1}: {conf:.2f}"
                cv2.putText(original_img, label_text, (box[0], box[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # --- 绘制关键点 ---
                if kpts is not None and kpts.size > 0:
                    for kp in kpts:
                        x, y = int(kp[0]), int(kp[1])
                        if x > 0 and y > 0:
                            cv2.circle(original_img, (x, y), 6, (0, 0, 255), -1)

        # 保存结果图
        save_path = os.path.join(save_dir, f"top2_{img_file}")
        cv2.imwrite(save_path, original_img)

    print(f"\n✅ Done! Check {len(image_files)} random results in '{save_dir}' folder.")


if __name__ == '__main__':
    run_top2_anchors_inference()