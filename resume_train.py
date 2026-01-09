import os
import torch

# --- 显存优化 ---
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

from ultralytics import YOLO


def train_final_v4():
    # 1. 加载权重
    # 指定加载你上一次训练最好的权重
    ckpt_path = r"F:\Spectral Scale Inspection\chart_training\mixed_L_Final(third)\weights\best.pt"

    print(f"Loading checkpoint from {ckpt_path}...")
    model = YOLO(ckpt_path)

    # 2. 开始最终阶段训练
    model.train(
        # =====================================================
        # [核心路径] 指向 prepare_dataset_v4.py 生成的新配置
        # =====================================================
        data=r"F:\Spectral Scale Inspection\dataset_v8_final\chart_pose_v4.yaml",

        # --- 显存/内存保命配置 (Windows 稳定版) ---
        batch=2,
        imgsz=960,  # 保持高分辨率以识别微小刻度
        workers=0,  # 必须为0，防止 I/O Error
        cache=False,

        # --- 基础配置 ---
        resume=False,  # 我们是加载权重微调，不是断点续训，所以设为 False

        # 【策略调整 1】轮数增加
        # 引入了双Y轴和大量真图，模型需要更多时间消化
        epochs=40,

        device=0,
        amp=True,

        # --- 目录与命名 ---
        project='chart_training',
        name='mixed_L_Final_v4_DoubleY(fourth)',  # 标记这是包含双Y轴的最终版

        # --- 优化器 (微调模式) ---
        optimizer='AdamW',
        lr0=0.0002,  # 保持低学习率，保护已有知识
        lrf=0.05,
        cos_lr=True,
        warmup_epochs=2,  # 给新数据一点热身时间

        # --- 正则化 ---
        dropout=0.1,
        weight_decay=0.001,  # 保持较强正则化，防止过拟合

        # --- Loss 权重 (激进提升召回率) ---
        # 【策略调整 2】加大 Box 权重
        # 为了解决漏检（特别是右侧 Y 轴容易被漏掉），大幅提升框的惩罚力度
        box=8.5,
        pose=12.0,  # 保持高精度关键点权重
        dfl=1.5,

        # --- 增强策略 (双Y轴适配版) ---
        # 1. 几何变换
        degrees=15.0,  # 应对 X 轴倾斜字体
        translate=0.1,
        scale=0.8,  # 大尺度波动
        shear=2.0,  # 模拟扫描畸变

        # 【策略调整 3】双 Y 轴专用设置
        fliplr=0.0,  # ❌ 严禁左右翻转！左轴变右轴会由逻辑错误。
        flipud=0.0,
        perspective=0.0005,  # ✅ 增加透视，模拟书本弯曲（对双轴识别有帮助）

        # 2. 混合增强
        hsv_h=0.015, hsv_s=0.6, hsv_v=0.5,
        mosaic=1.0,
        mixup=0.25,  # 深度混合真假数据
        copy_paste=0.1,
        erasing=0.4,  # 模拟遮挡

        close_mosaic=10,  # 【策略调整 4】最后 10 轮关闭增强，让模型在纯净数据上稳一稳

        save=True, exist_ok=True, plots=True, val=True
    )
    print("✅ Final V4 Training (Double Y-Axis Supported) Completed!")


if __name__ == '__main__':
    train_final_v4()