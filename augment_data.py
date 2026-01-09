import albumentations as A
import cv2
import os
from tqdm import tqdm

# ================= 配置区域 =================
# 1. 输入路径 (你的合成数据集原图位置)
INPUT_ROOT = r"F:\Spectral Scale Inspection\dataset_v8_final\synthetic images"

# 2. 输出路径 (脚本会自动创建这个文件夹，和 synthetic images 同级)
# 结果会保存在: F:\Spectral Scale Inspection\dataset_v8_final\synthetic images_degraded
OUTPUT_ROOT = r"F:\Spectral Scale Inspection\dataset_v8_final\synthetic images_degraded"

# 支持的图片扩展名
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
# ===========================================

# --- 定义画质“腐蚀”增强流 ---
# 仅做像素增强，不改变几何形状，因此你可以直接复制标签文件，无需重新标注
transform = A.Compose([
    # 1. 模拟模糊 (高斯模糊或动态模糊)
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 7), p=1),
        A.MotionBlur(blur_limit=(3, 7), p=1),
    ], p=0.5),

    # 2. 模拟 JPG 压缩伪影 (这是图表 Sim2Real 最关键的一步)
    A.ImageCompression(quality_lower=30, quality_upper=75, p=0.6),

    # 3. 添加噪点 (模拟扫描件噪点)
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),

    # 4. 颜色略微抖动 (防止过拟合纯色)
    A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.01, p=0.3),

    # 5. 随机转灰度 (部分真实论文是黑白打印的)
    A.ToGray(p=0.15),
])


def process_images():
    print(f"🚀 开始处理...")
    print(f"源目录: {INPUT_ROOT}")
    print(f"目标目录: {OUTPUT_ROOT}")

    count = 0

    # 使用 os.walk 递归遍历所有子文件夹 (train, val 等)
    for root, dirs, files in os.walk(INPUT_ROOT):
        for file in tqdm(files, desc=f"Scanning {os.path.basename(root)}"):
            if os.path.splitext(file)[1].lower() in EXTENSIONS:
                # 构建完整源文件路径
                src_path = os.path.join(root, file)

                # 计算相对路径 (例如: train/001.jpg)
                rel_path = os.path.relpath(src_path, INPUT_ROOT)

                # 构建完整目标文件路径
                dst_path = os.path.join(OUTPUT_ROOT, rel_path)

                # 确保目标文件夹存在
                dst_dir = os.path.dirname(dst_path)
                if not os.path.exists(dst_dir):
                    os.makedirs(dst_dir)

                # --- 核心处理 ---
                image = cv2.imread(src_path)
                if image is None:
                    print(f"⚠️ 无法读取: {src_path}")
                    continue

                # 应用增强
                try:
                    augmented = transform(image=image)["image"]
                    # 保存图片
                    cv2.imwrite(dst_path, augmented)
                    count += 1
                except Exception as e:
                    print(f"❌ 处理出错 {src_path}: {e}")

    print("-" * 30)
    print(f"✅ 处理完成！共生成 {count} 张“腐蚀”图片。")
    print(f"📂 输出位置: {OUTPUT_ROOT}")
    print("-" * 30)
    print("⚠️ 【重要提示】下一步操作：")
    print("1. 请手动复制对应的 labels 文件夹。")
    print("   例如：将 labels/train 复制为 labels_degraded/train")
    print("2. 修改 yaml 文件中的 train 路径指向新的 synthetic images_degraded 文件夹。")


if __name__ == "__main__":
    process_images()