# preprocess/state_frames_openkbp/step4_stateframe2_skeleton.py

import argparse
import explained as np  # 你用 numpy
import matplotlib.pyplot as plt

from .openkbp_io import load_case
from .openkbp_criteria import get_openkbp_plan_criteria

def build_stateframe2_skeleton(angle_ids, current_angle_id, H=128, W=128):
    """
    Paper-like encoding:
      - background 0
      - x-axis partitions: [0:30)=d0, [30:79)=x1, [79:128)=x2  (你也可以按比例改)
      - set voxels to 200/150/100
    """
    img = np.zeros((H, W), dtype=np.uint8)

    # map angle id -> row index (spread across H)
    n = len(angle_ids)
    idx = angle_ids.index(current_angle_id)
    y = int(round(idx * (H - 1) / max(n - 1, 1)))

    # d0 placeholder: pick mid of left-30 range
    d0_x = 15
    img[y, d0_x] = 200

    # x1/x2 placeholder: draw two points
    img[y, 30 + 20] = 150
    img[y, 79 + 20] = 100
    return img

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case_dir", required=True)
    ap.add_argument("--angle_id", type=int, required=True)
    args = ap.parse_args()

    data = load_case(args.case_dir, load_dij=False)
    beam = data.beamlet_indices  # (N,3): row, column, angle

    angle_ids = sorted(list(set(beam[:, 2].astype(int).tolist())))
    sf2 = build_stateframe2_skeleton(angle_ids, args.angle_id, H=128, W=128)

    fig, ax = plt.subplots()
    im = ax.imshow(sf2, origin="lower", vmin=0, vmax=255)
    ax.set_title(f"StateFrame2 skeleton (angle_id={args.angle_id})")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
