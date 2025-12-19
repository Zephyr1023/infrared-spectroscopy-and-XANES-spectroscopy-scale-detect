from ultralytics import YOLO


def train():
    # 1. 加载模型
    print("Loading YOLO11-Large Pose model...")
    model = YOLO('yolo11l-pose.pt')

    # 2. 开始训练
    model.train(
        # --- 基础配置 ---
        data='chart_pose.yaml',
        epochs=30,  # [提升] 30
        imgsz=1024,  # [核心提升] 640 -> 1024。图表细节很小，大分辨率对精度提升最明显！
        # 警告：1024分辨率下，显存压力大增，batch必须调小

        # --- 硬件与显存优化 (RTX 5070 Ti Laptop) ---
        batch=4,  # [注意] 如果imgsz=1024，这里可能要降到 4 或 6。如果爆显存就改小。
        device=0,
        workers=4,
        amp=True,  # 混合精度，必须开

        project='chart_training',
        name='exp_pose_large_enhanced',

        # --- 优化器与学习率策略 (针对合成数据的优化) ---
        optimizer='AdamW',  # [提升] AdamW 对这种规律性强的数据通常收敛更好
        lr0=0.001,  # 初始学习率 (配合 AdamW 稍微调小一点，默认是 0.01)
        cos_lr=True,  # [提升] 使用余弦退火学习率，后期能更好地收敛到最优解
        warmup_epochs=3,  # 预热 3 轮

        # --- 数据增强策略 (专门针对图表的调整) ---
        # 1. 几何增强：图表通常是正的，不需要大幅旋转
        degrees=0.5,  # 仅允许极微小的旋转 (+/- 0.5度)
        perspective=0.000,  # [禁用] 透视变换会扭曲坐标轴，导致直线变弯，对图表识别有害
        shear=0.0,  # [禁用] 剪切变换

        # 2. 颜色与噪声：模拟真实论文的扫描/截图质量
        hsv_h=0.015,  # 色调变化微调
        hsv_s=0.4,  # 饱和度变化
        hsv_v=0.4,  # 亮度变化 (模拟不同光照/截图背景)

        # 3. 核心增强算法
        mosaic=1.0,  # [保留] Mosaic 非常重要，能让模型学习不同尺度的目标
        mixup=0.0,  # [禁用] MixUp 会把两张图表叠在一起，这对文字识别是毁灭性的干扰
        copy_paste=0.0,  # [禁用] 同样不适合 Pose 任务

        # 4. 高级策略：最后阶段关闭 Mosaic
        # 这会让最后 10 轮训练完全使用真实（非拼接）的图片，大幅提升检测精度
        close_mosaic=10,

        # --- 其他 ---
        save=True,
        exist_ok=True,
        plots=True,  # 训练结束后自动画出混淆矩阵等图表
    )
    print("✅ 增强训练完成！")


if __name__ == '__main__':
    train()