import os
import shutil
import glob
import random
from tqdm import tqdm

# ================= 配置区域 =================

# 1. 真实数据源 (输入)
REAL_SOURCE_IMG_DIR = r"/Fourth Raw Dataset"
REAL_SOURCE_TXT_DIR = r"/Fourth Raw Dataset_txt"

# 2. 合成数据源 (输入 - 假设在当前项目目录下)
SYNTH_IMG_DIR = r"dataset_v8_final/synthetic images_degraded"
SYNTH_LBL_DIR = r"dataset_v8_final/synthetic labels_pose"

# 3. 目标输出根目录
BASE_OUTPUT_DIR = r"F:\Spectral Scale Inspection\dataset_v8_final"

# 4. 划分策略 (8:1:1)
SPLIT_RATIO = {
    'train': 0.8,
    'val': 0.1,
    'test': 0.1
}

# 5. 真实训练数据过采样倍数
# 700张真图 -> 训练集约560张 -> 3倍复制 -> 1680张真图混入训练集
OVERSAMPLE_FACTOR = 3


# ===========================================

def setup_dirs():
    """清理并创建特定的 train4, val4, test4 目录结构"""
    # 定义三个目标文件夹名
    dirs = ['train4', 'val4', 'test4']

    for d in dirs:
        target_path = os.path.join(BASE_OUTPUT_DIR, d)

        # 如果文件夹存在，先清理内部的 images 和 labels，保留文件夹本身以防权限问题
        # 或者简单粗暴点：直接重建
        if os.path.exists(target_path):
            print(f"🧹 清理旧目录: {target_path}")
            shutil.rmtree(target_path)

        # 创建 images 和 labels 子目录 (YOLO 标准结构)
        os.makedirs(os.path.join(target_path, 'images'), exist_ok=True)
        os.makedirs(os.path.join(target_path, 'labels'), exist_ok=True)


def get_real_data_pairs():
    """获取所有配对成功的真实图片和txt路径"""
    print(f"🔍 正在扫描真实数据: {REAL_SOURCE_IMG_DIR} ...")

    # 支持的图片格式
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
    img_paths = []
    for ext in extensions:
        # 递归查找或单层查找，这里用单层
        img_paths.extend(glob.glob(os.path.join(REAL_SOURCE_IMG_DIR, ext)))

    valid_pairs = []
    for img_path in img_paths:
        basename = os.path.splitext(os.path.basename(img_path))[0]
        txt_path = os.path.join(REAL_SOURCE_TXT_DIR, basename + ".txt")

        if os.path.exists(txt_path):
            valid_pairs.append((img_path, txt_path))
        else:
            # 可选：打印缺失标签的图片
            pass

    # 随机打乱
    random.shuffle(valid_pairs)
    print(f"✅ 找到 {len(valid_pairs)} 组有效真实数据 (图片+TXT)")
    return valid_pairs


def copy_files_to_folder(pairs, folder_name, is_real_train=False, description=""):
    """
    将文件复制到 train4/val4/test4 下的 images 和 labels
    """
    target_img_dir = os.path.join(BASE_OUTPUT_DIR, folder_name, 'images')
    target_lbl_dir = os.path.join(BASE_OUTPUT_DIR, folder_name, 'labels')

    desc = f"[{description}] -> {folder_name}"
    if is_real_train:
        desc += f" (过采样 {OVERSAMPLE_FACTOR} 倍)"

    print(f"\n🚀 正在处理 {desc} ...")

    count = 0
    for img_path, txt_path in tqdm(pairs):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        ext = os.path.splitext(img_path)[1]

        # 确定复制次数
        repeat = OVERSAMPLE_FACTOR if is_real_train else 1

        for i in range(repeat):
            # 生成新文件名
            if is_real_train:
                # 训练集：加后缀避免重名，且制造多份副本
                new_base_name = f"{base_name}_real_{i}"
            else:
                # 验证/测试集：保持原名或简单后缀，不需要复制
                new_base_name = base_name

            dst_img = os.path.join(target_img_dir, new_base_name + ext)
            dst_txt = os.path.join(target_lbl_dir, new_base_name + ".txt")

            shutil.copy(img_path, dst_img)
            shutil.copy(txt_path, dst_txt)
            count += 1

    print(f"   已生成 {count} 个文件到 {folder_name}")


def copy_synthetic_data():
    """搬运合成数据 -> 只能去 train4"""
    print(f"\n🚀 处理 [合成数据] -> train4 ...")

    target_img_dir = os.path.join(BASE_OUTPUT_DIR, 'train4', 'images')
    target_lbl_dir = os.path.join(BASE_OUTPUT_DIR, 'train4', 'labels')

    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    img_files = []
    for ext in extensions:
        img_files.extend(glob.glob(os.path.join(SYNTH_IMG_DIR, ext)))

    count = 0
    for img_path in tqdm(img_files):
        basename = os.path.splitext(os.path.basename(img_path))[0]
        txt_path = os.path.join(SYNTH_LBL_DIR, basename + ".txt")

        if os.path.exists(txt_path):
            # 合成数据不需要改名，直接复制
            dst_img = os.path.join(target_img_dir, os.path.basename(img_path))
            dst_txt = os.path.join(target_lbl_dir, basename + ".txt")

            shutil.copy(img_path, dst_img)
            shutil.copy(txt_path, dst_txt)
            count += 1

    print(f"✅ 合成数据搬运完成: {count} 张")


def create_yaml():
    """生成指向 train4, val4, test4 的 yaml 文件"""

    # 构造绝对路径，确保 YOLO 能找到
    path_train = os.path.join(BASE_OUTPUT_DIR, 'train4', 'images')
    path_val = os.path.join(BASE_OUTPUT_DIR, 'val4', 'images')
    path_test = os.path.join(BASE_OUTPUT_DIR, 'test4', 'images')

    yaml_content = f"""
# YOLOv11 Chart Pose Config (Custom Split: train4/val4/test4)

# Absolute paths
train: {path_train}
val: {path_val}
test: {path_test}

# Keypoints definition (x, y, visible)
kpt_shape: [2, 3] 

# Classes
names:
  0: x_axis
  1: y_axis
"""
    # yaml 保存到 dataset_v8_final 根目录下
    yaml_path = os.path.join(BASE_OUTPUT_DIR, "chart_pose_v4.yaml")
    with open(yaml_path, "w", encoding='utf-8') as f:
        f.write(yaml_content)
    print(f"\n📄 YAML 配置文件已生成: {yaml_path}")
    return yaml_path


if __name__ == "__main__":
    setup_dirs()

    # 1. 获取并切分真实数据
    all_real_pairs = get_real_data_pairs()
    total = len(all_real_pairs)

    if total > 0:
        # 计算切分点
        n_train = int(total * SPLIT_RATIO['train'])
        n_val = int(total * SPLIT_RATIO['val'])

        train_pairs = all_real_pairs[:n_train]
        val_pairs = all_real_pairs[n_train: n_train + n_val]
        test_pairs = all_real_pairs[n_train + n_val:]

        print(f"📊 真实数据划分统计: 总数 {total}")
        print(f"   Train (train4): {len(train_pairs)} (将执行 {OVERSAMPLE_FACTOR}倍 过采样)")
        print(f"   Val   (val4)  : {len(val_pairs)}  (纯真实)")
        print(f"   Test  (test4) : {len(test_pairs)}  (纯真实)")

        # 2. 执行复制
        # Train4: 真实数据过采样
        copy_files_to_folder(train_pairs, 'train4', is_real_train=True, description="真实训练集")

        # Val4: 纯真实，不过采样
        copy_files_to_folder(val_pairs, 'val4', is_real_train=False, description="真实验证集")

        # Test4: 纯真实，不过采样
        copy_files_to_folder(test_pairs, 'test4', is_real_train=False, description="真实测试集")

    else:
        print("❌ 错误：未找到真实数据，请检查路径配置！")

    # 3. 将合成数据全部放入 train4
    if os.path.exists(SYNTH_IMG_DIR):
        copy_synthetic_data()
    else:
        print(f"⚠️ 警告：找不到合成数据目录 {SYNTH_IMG_DIR}，跳过合成数据复制。")

    # 4. 生成配置
    final_yaml = create_yaml()

    print("\n🎉 数据集准备完毕！")
    print(f"训练代码中请设置: data=r'{final_yaml}'")