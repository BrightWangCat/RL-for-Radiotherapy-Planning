import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy.ndimage import rotate

# =================配置区域=================
CASE_PATH = "/fs/scratch/PCON0023/mingshiw/PlanData/open-kbp-opt-data/reference-plans/pt_241"
OUTPUT_DIR = "./output_step3_state2"
FULL_SHAPE = (128, 128, 128)
TOTAL_VOXELS = 128 * 128 * 128

# 论文设定参数
NUM_ANGLES = 96  # 一圈 96 个控制点
GRID_SIZE = 96   # 网络输入大小 96x96
PTV_MARGIN_MM = 10.0 # 1cm margin
VOXEL_SIZE_MM = 3.4  # OpenKBP 体素大概是 3mm-4mm，我们需要读取真实值，这里先预设
# =========================================

def load_sparse_csv(csv_path, is_structure=False):
    """(复用读取代码)"""
    if not os.path.exists(csv_path): return None
    try:
        df = pd.read_csv(csv_path)
        indices = pd.to_numeric(df.iloc[:, 0], errors='coerce').fillna(-1).astype(int)
        valid_mask = (indices >= 0) & (indices < TOTAL_VOXELS)
        indices = indices[valid_mask]
        dense_flat = np.zeros(TOTAL_VOXELS, dtype=np.float32)
        if is_structure:
            dense_flat[indices] = 1.0
        else:
            values = pd.to_numeric(df.iloc[:, -1], errors='coerce').fillna(0.0)
            values = values[valid_mask]
            dense_flat[indices] = values
        return dense_flat.reshape(FULL_SHAPE)
    except Exception as e: return None

def get_voxel_size(case_path):
    # 尝试读取真实的体素尺寸
    dim_path = os.path.join(case_path, "voxel_dimensions.csv")
    if os.path.exists(dim_path):
        # 假设 csv 里存的是 x, y, z 的尺寸
        # OpenKBP: usually [dx, dy, dz]
        try:
            df = pd.read_csv(dim_path, header=None)
            vals = df.values.flatten()
            print(f"Voxel dimensions loaded: {vals}")
            # 只要取平面上的尺寸 (x or y)
            return vals[0] 
        except:
            pass
    print("Warning: Could not read voxel size, using default 3.0mm")
    return 3.0 # Default fallback

def calculate_conformal_sinogram(ptv_slice, voxel_size_mm):
    """
    计算某一层的 Conformal Arc Sinogram。
    核心逻辑：旋转 PTV -> 找左右边界 -> 映射到 96x96 图像
    """
    sinogram = np.zeros((NUM_ANGLES, GRID_SIZE), dtype=np.uint8)
    
    # 找到 PTV 的重心，用于居中处理 (RL通常以Isocenter为中心)
    # OpenKBP 数据通常已经是配准好的，Isocenter 在图像中心 (64, 64)
    # 所以我们只需要围绕中心旋转
    
    # 将 PTV 扩大 margin (膨胀操作，或者简单地在找边界时加 margin)
    # 这里我们在找边界时直接加像素宽度的 margin
    margin_pixels = int(PTV_MARGIN_MM / voxel_size_mm)
    
    # 遍历所有角度
    for i, angle in enumerate(np.linspace(0, 360, NUM_ANGLES, endpoint=False)):
        # 1. 旋转 PTV Mask
        # scipy.ndimage.rotate: counter-clockwise, so we use -angle to simulate gantry
        # reshape=False 保证图像大小不变
        rotated_mask = rotate(ptv_slice, -angle, reshape=False, order=0, mode='constant', cval=0)
        
        # 2. 投影到 X 轴 (Beam's Eye View 的 1D 切面)
        # 在旋转后的坐标系中，射线源在上方，射向下方。MLC 沿着 X 轴左右运动遮挡。
        # 只要这一列有 PTV (sum > 0)，就说明这里需要由 MLC 打开
        projection = np.any(rotated_mask, axis=0) # 投影到X轴
        
        # 3. 找到左右边界 (Leaf positions)
        indices = np.where(projection)[0]
        
        if len(indices) > 0:
            x1_raw = indices[0] - margin_pixels  # Left leaf edge
            x2_raw = indices[-1] + margin_pixels # Right leaf edge
            
            # 4. 坐标映射: 128 (原始) -> 96 (网络输入)
            # 假设物理中心对齐。
            scale_ratio = GRID_SIZE / FULL_SHAPE[0] # 96 / 128 = 0.75
            
            x1_net = int(x1_raw * scale_ratio)
            x2_net = int(x2_raw * scale_ratio)
            
            # 边界检查
            x1_net = np.clip(x1_net, 0, GRID_SIZE-1)
            x2_net = np.clip(x2_net, 0, GRID_SIZE-1)
            
            # 5. 画图 (State Frame 2 定义)
            # x1 leaf: value 150
            sinogram[i, x1_net] = 150
            # x2 leaf: value 100
            sinogram[i, x2_net] = 100
            
            # Dose Rate: value 200
            # 初始状态通常设为最大剂量率或常数，论文说 "dose rate remained constant"
            # 我们假设它在某个固定位置，比如 x=10 (为了可视化) 或者根据论文 Fig 2b 那样的一条线
            # 论文图里 Dose rate 线是直的
            d_rate_pos = 10 # 假定位置
            sinogram[i, d_rate_pos] = 200
            
        else:
            # 如果这一层在这个角度看不到 PTV (比如 PTV 形状不规则)，MLC 闭合
            mid = GRID_SIZE // 2
            sinogram[i, mid] = 150
            sinogram[i, mid] = 100 # 重叠即闭合

    return sinogram

def main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    print(f"Processing case: {CASE_PATH}")

    # 1. 读取数据
    ptv_files = glob.glob(os.path.join(CASE_PATH, "PTV*.csv"))
    if not ptv_files: return
    ptv_path = sorted(ptv_files)[-1]
    ptv_mask = load_sparse_csv(ptv_path, is_structure=True)
    
    if ptv_mask is None: return

    # 读取真实体素尺寸
    voxel_size = get_voxel_size(CASE_PATH)
    print(f"Using voxel size: {voxel_size} mm")

    # 2. 选择一个切片进行计算
    coords = np.argwhere(ptv_mask > 0)
    z_slice = int(np.mean(coords[:, 0])) if len(coords) > 0 else 64
    print(f"Computing State Frame 2 for Slice z={z_slice}")
    
    ptv_slice = ptv_mask[z_slice, :, :] # 取出 2D 切面

    # 3. 计算 Sinogram
    state_frame_2 = calculate_conformal_sinogram(ptv_slice, voxel_size)

    # 4. 可视化
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # 左图: PTV Slice (看看我们基于什么算的)
    axes[0].set_title(f"PTV Slice (z={z_slice})")
    axes[0].imshow(ptv_slice, cmap='gray')
    
    # 右图: State Frame 2 (Sinogram)
    axes[1].set_title("State Frame 2 (Init Conformal Arc)")
    # 使用 jet 或 gray 都可以，只要能区分 100, 150, 200
    im = axes[1].imshow(state_frame_2, cmap='nipy_spectral', aspect='auto', vmin=0, vmax=255)
    plt.colorbar(im, ax=axes[1], label='Pixel Value (100=Leaf2, 150=Leaf1, 200=DoseRate)')
    
    # 标注轴
    axes[1].set_xlabel("Leaf/Jaw Position (Scaled to 96)")
    axes[1].set_ylabel("Gantry Angle (0-360)")

    save_path = os.path.join(OUTPUT_DIR, "step3_state2_visualization.png")
    plt.savefig(save_path)
    print(f"Visualization saved to: {save_path}")

if __name__ == "__main__":
    main()