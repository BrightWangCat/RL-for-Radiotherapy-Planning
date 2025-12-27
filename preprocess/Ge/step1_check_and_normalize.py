import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob

# =================配置区域=================
CASE_PATH = "/fs/scratch/PCON0023/mingshiw/PlanData/open-kbp-opt-data/reference-plans/pt_241"
OUTPUT_DIR = "./output_step1_result"
FULL_SHAPE = (128, 128, 128)
TOTAL_VOXELS = 128 * 128 * 128
# =========================================

def load_sparse_csv(csv_path, is_structure=False):
    """
    读取稀疏CSV。
    is_structure=True: 忽略数值列，所有存在的索引都置为1。
    is_structure=False: 使用数值列 (用于Dose)。
    """
    if not os.path.exists(csv_path):
        print(f"Error: File not found {csv_path}")
        return None
    
    try:
        df = pd.read_csv(csv_path)
        # 第一列是索引
        indices = pd.to_numeric(df.iloc[:, 0], errors='coerce').fillna(-1).astype(int)
        
        # 过滤有效索引
        valid_mask = (indices >= 0) & (indices < TOTAL_VOXELS)
        indices = indices[valid_mask]
        
        # 初始化 3D 矩阵
        dense_flat = np.zeros(TOTAL_VOXELS, dtype=np.float32)

        if is_structure:
            # 策略修改：如果是结构，直接将这些索引位置置 1
            dense_flat[indices] = 1.0
            print(f"Loaded structure {os.path.basename(csv_path)}: {len(indices)} voxels set to 1.")
        else:
            # 如果是剂量，读取最后一列的值
            values = pd.to_numeric(df.iloc[:, -1], errors='coerce').fillna(0.0)
            values = values[valid_mask]
            dense_flat[indices] = values
            print(f"Loaded dose {os.path.basename(csv_path)}: Max value {np.max(values)}")

        return dense_flat.reshape(FULL_SHAPE)

    except Exception as e:
        print(f"Error processing {csv_path}: {e}")
        return None

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"Processing case: {CASE_PATH}")

    # 1. 读取 Dose
    dose_path = os.path.join(CASE_PATH, "dose.csv")
    dose_grid = load_sparse_csv(dose_path, is_structure=False)

    # 2. 读取 PTV (强制模式 is_structure=True)
    # 自动寻找最大的PTV文件
    ptv_files = glob.glob(os.path.join(CASE_PATH, "PTV*.csv"))
    if not ptv_files:
        print("Error: No PTV file found.")
        return
    ptv_path = sorted(ptv_files)[-1] # 选最大的，通常是PTV70
    print(f"Using PTV file: {ptv_path}")
    
    ptv_mask = load_sparse_csv(ptv_path, is_structure=True)

    if dose_grid is None or ptv_mask is None:
        return

    # 3. 归一化计算
    # 论文: set maximum PTV dose equal to 1.08
    ptv_indices = ptv_mask > 0.5 # 既然全是1，用 >0.5 即可
    
    if not np.any(ptv_indices):
        print("Error: PTV mask is still empty. Check file content manually.")
        return

    max_ptv_dose = np.max(dose_grid[ptv_indices])
    print(f"Max Dose in PTV ROI: {max_ptv_dose} Gy")

    scaling_factor = 1.08 / max_ptv_dose
    print(f"Scaling Factor: {scaling_factor}")
    
    normalized_dose = dose_grid * scaling_factor
    print(f"New Max PTV Dose (Target 1.08): {np.max(normalized_dose[ptv_indices])}")

    # 保存一下归一化后的剂量矩阵，后面步骤可能要用
    # np.save(os.path.join(OUTPUT_DIR, "normalized_dose.npy"), normalized_dose)
    # np.save(os.path.join(OUTPUT_DIR, "ptv_mask.npy"), ptv_mask)

    # 4. 可视化
    # 选取包含 PTV 重心的切片，这样一定能看到东西
    coords = np.argwhere(ptv_mask > 0)
    z_center = int(np.mean(coords[:, 0])) # 假设第一维是Z? 还是需要尝试
    # OpenKBP 形状 (128, 128, 128)，我们取三个方向的中值
    z_slice = int(np.mean(coords[:, 0])) if len(coords) > 0 else 64
    y_slice = int(np.mean(coords[:, 1])) if len(coords) > 0 else 64
    x_slice = int(np.mean(coords[:, 2])) if len(coords) > 0 else 64

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # 左上: PTV Mask (Axial)
    axes[0,0].set_title(f"PTV Mask (Slice {z_slice})")
    axes[0,0].imshow(ptv_mask[z_slice, :, :], cmap='gray')

    # 右上: Original Dose
    axes[0,1].set_title(f"Original Dose (Max={max_ptv_dose:.2f})")
    axes[0,1].imshow(dose_grid[z_slice, :, :], cmap='jet')
    
    # 左下: Normalized Dose
    axes[1,0].set_title("Normalized Dose (PTV Max=1.08)")
    im = axes[1,0].imshow(normalized_dose[z_slice, :, :], cmap='jet', vmin=0, vmax=1.2)
    plt.colorbar(im, ax=axes[1,0])

    # 右下: 直方图 (验证数值分布)
    axes[1,1].set_title("Dose Histogram inside PTV")
    axes[1,1].hist(normalized_dose[ptv_indices].flatten(), bins=50, color='red', alpha=0.7)
    axes[1,1].axvline(1.08, color='black', linestyle='dashed', linewidth=2, label='1.08 Limit')
    axes[1,1].legend()

    save_path = os.path.join(OUTPUT_DIR, "step1_final_check.png")
    plt.savefig(save_path)
    print(f"Visualization saved to: {save_path}")

if __name__ == "__main__":
    main()