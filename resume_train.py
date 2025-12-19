import os
import torch

# --- 显存优化 ---
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

from ultralytics import YOLO


def train_sim2real_mixed():
    # 1. 加载权重
    # 继续沿用你之前训练出的最好的权重作为起点
    ckpt_path = r"chart_training/sim2real_L_stable/weights/best.pt"

    # 如果找不到上面的，可以用 optimized 版本的
    if not os.path.exists(ckpt_path):
        ckpt_path = r"chart_training/sim2real_L_optimized(second)/weights/best.pt"

    print(f"Loading checkpoint from {ckpt_path}...")
    model = YOLO(ckpt_path)

    # 2. 开始混合训练
    model.train(
        # --- 核心路径 ---
        data='yolo_chart_dataset/chart_pose_mixed.yaml',  # 【注意】确保这里是 mixed 的 yaml

        # --- 显存/内存保命配置 (绝对不改) ---
        batch=2,
        imgsz=960,
        workers=0,
        cache=False,

        # --- 基础配置 ---
        resume=False,
        epochs=40,  # 【调整 1】混合数据需要更多轮次来消化
        device=0,
        amp=True,

        # --- 目录与命名 ---
        project='chart_training',
        name='mixed_L_Final(third)',  # 改名：这是最终的混合训练版本

        # --- 优化器 ---
        optimizer='AdamW',
        lr0=0.0002,  # 保持低学习率微调
        lrf=0.05,
        cos_lr=True,
        warmup_epochs=2,

        # --- 正则化 ---
        dropout=0.1,  # 保持 Dropout 防止过拟合
        weight_decay=0.001,

        # --- Loss 权重调整 (解决漏检) ---
        box=7.5,  # 【调整 2】恢复到默认 7.5 (之前是5.0)，强迫模型找回漏掉的框
        pose=12.0,  # 【调整 3】稍微降低一点 (之前是15.0)，给 Box 让一点路
        dfl=1.5,

        # --- 增强策略 (Sim2Real 进攻型配置) ---
        # 1. 针对 X 轴文字旋转
        degrees=15.0,  # 【调整 4】加大旋转！真实图表 X 轴经常有斜字

        # 2. 针对尺寸和位置
        translate=0.1,
        scale=0.8,  # 【调整 5】加大尺寸波动 (0.2 ~ 1.8)，适应不同分辨率
        shear=2.0,  # 【调整 6】加大剪切，模拟扫描畸变
        perspective=0.0005,

        # 3. 颜色与混合
        hsv_h=0.015, hsv_s=0.6, hsv_v=0.5,
        fliplr=0.0, flipud=0.0,  # 严禁翻转

        mosaic=1.0,
        mixup=0.25,  # 【调整 7】加大 Mixup (之前0.15)，深度融合真假数据
        copy_paste=0.1,
        erasing=0.4,  # 模拟遮挡

        close_mosaic=5,  # 最后 5 轮关闭增强，做精细收尾

        save=True, exist_ok=True, plots=True, val=True
    )
    print("✅ Mixed Dataset Training Completed!")


if __name__ == '__main__':
    train_sim2real_mixed()