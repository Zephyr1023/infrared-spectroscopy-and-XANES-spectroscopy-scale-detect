import os
import shutil
import glob
from tqdm import tqdm

# ================= 配置区域 =================
# 1. 训练集 A (合成数据 - 主力军)
SYNTH_IMG_DIR = "dataset_v8_final/synthetic images_degraded"
SYNTH_LBL_DIR = "dataset_v8_final/synthetic labels_pose"

# 2. 训练集 B (真实数据 - 特种部队，需增强)
# 这部分数据会被复制 5 份混入训练集
REAL_TRAIN_IMG_DIR = r"dataset_v8_final/train_real image and json"
REAL_TRAIN_LBL_DIR = r"dataset_v8_final/train_real txt"
OVERSAMPLE_FACTOR = 5  # 复制倍数

# 3. 验证集 (真实数据 - 保持纯净)
REAL_VAL_IMG_DIR = r"dataset_v8_final/val_real image and json"
REAL_VAL_LBL_DIR = r"dataset_v8_final/val_real txt txt"

# 4. 目标输出目录
DEST_DIR = "yolo_chart_dataset"


# ===========================================

def setup_dirs():
    """创建标准的 YOLO 目录结构"""
    if os.path.exists(DEST_DIR):
        print(f"🧹 清理旧目录: {DEST_DIR}")
        shutil.rmtree(DEST_DIR)

    # 创建 synthetic images/train, synthetic images/val, labels/train, labels/val
    for split in ['train', 'val']:
        os.makedirs(os.path.join(DEST_DIR, 'synthetic images', split), exist_ok=True)
        os.makedirs(os.path.join(DEST_DIR, 'labels', split), exist_ok=True)


def copy_data_group(src_img_dir, src_lbl_dir, split_type, description):
    """
    通用复制函数 (用于合成数据和验证集)
    """
    print(f"\n🚀 正在处理 [{description}] -> {split_type} ...")

    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    img_files = []
    for ext in extensions:
        img_files.extend(glob.glob(os.path.join(src_img_dir, ext)))

    count = 0
    missing_labels = 0

    for img_path in tqdm(img_files):
        img_name = os.path.basename(img_path)
        txt_name = os.path.splitext(img_name)[0] + ".txt"
        src_txt_path = os.path.join(src_lbl_dir, txt_name)

        if os.path.exists(src_txt_path):
            dst_img_path = os.path.join(DEST_DIR, 'synthetic images', split_type, img_name)
            dst_txt_path = os.path.join(DEST_DIR, 'labels', split_type, txt_name)

            shutil.copy(img_path, dst_img_path)
            shutil.copy(src_txt_path, dst_txt_path)
            count += 1
        else:
            missing_labels += 1

    print(f"✅ [{description}] 处理完成: 成功复制 {count} 张。")


def copy_real_train_augmented(src_img_dir, src_lbl_dir, factor):
    """
    【新增】专门处理真实训练集，执行“过采样”策略 (复制 N 份)
    """
    print(f"\n🚀 正在处理 [真实训练数据] -> train (过采样 {factor} 倍) ...")

    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    img_files = []
    for ext in extensions:
        img_files.extend(glob.glob(os.path.join(src_img_dir, ext)))

    count = 0
    total_generated = 0
    missing_labels = 0

    for img_path in tqdm(img_files):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        ext = os.path.splitext(img_path)[1]

        # 对应的标签路径
        src_txt_path = os.path.join(src_lbl_dir, base_name + ".txt")

        if os.path.exists(src_txt_path):
            count += 1
            # 循环复制 N 次，每次改个名字防止覆盖
            for i in range(factor):
                # 生成新文件名：原名_copy_0.jpg, 原名_copy_1.jpg ...
                new_img_name = f"{base_name}_real_copy_{i}{ext}"
                new_txt_name = f"{base_name}_real_copy_{i}.txt"

                # 目标路径
                dst_img_path = os.path.join(DEST_DIR, 'synthetic images', 'train', new_img_name)
                dst_txt_path = os.path.join(DEST_DIR, 'labels', 'train', new_txt_name)

                # 复制
                shutil.copy(img_path, dst_img_path)
                shutil.copy(src_txt_path, dst_txt_path)
                total_generated += 1
        else:
            missing_labels += 1

    print(f"✅ [真实训练数据] 原始 {count} 张 -> 生成 {total_generated} 张 (混入训练集)")


def create_yaml():
    """自动生成配套的 .yaml 文件"""
    # 注意：这里我帮你把 kpt_shape 改成了 [2, 3]，防止你之前遇到的报错再次发生
    yaml_content = f"""
# YOLOv11 Chart Pose Config (Mixed Sim2Real)

path: {os.path.abspath(DEST_DIR)} # 数据集根目录
train: synthetic images/train  # 混合了合成数据 + 5倍真实数据
val: synthetic images/val      # 纯真实数据
test: synthetic images/val

# Keypoints definition
kpt_shape: [2, 3] # [关键点数量, 维度(x, y, visible)] 

# Classes
names:
  0: x_axis
  1: y_axis
"""
    yaml_path = os.path.join(DEST_DIR, "chart_pose_mixed.yaml")
    with open(yaml_path, "w", encoding='utf-8') as f:
        f.write(yaml_content)
    print(f"\n📄 YAML 配置文件已生成: {yaml_path}")


if __name__ == "__main__":
    setup_dirs()

    # 1. 搬运合成数据 (train)
    copy_data_group(SYNTH_IMG_DIR, SYNTH_LBL_DIR, 'train', "合成训练集")

    # 2. 搬运并增强真实训练数据 (train, 5倍)
    copy_real_train_augmented(REAL_TRAIN_IMG_DIR, REAL_TRAIN_LBL_DIR, OVERSAMPLE_FACTOR)

    # 3. 搬运真实验证数据 (val)
    copy_data_group(REAL_VAL_IMG_DIR, REAL_VAL_LBL_DIR, 'val', "真实验证集")

    # 4. 生成配置文件
    create_yaml()

    print("\n🎉 混合数据集准备完毕！")
    print(f"训练时请使用: data='{DEST_DIR}/chart_pose_mixed.yaml'")