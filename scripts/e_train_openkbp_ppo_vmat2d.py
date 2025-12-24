# scripts/e_train_openkbp_ppo_vmat2d.py
import os
import sys
import runpy

# Ensure env is registered before CleanRL calls gym.make()
import rlfplan.register_envs  # noqa: F401

PROJECT_ROOT = "/fs/scratch/PCON0023/mingshiw/RLfPlan5"
CLEANRL_SCRIPT = os.path.join(PROJECT_ROOT, "cleanrl", "cleanrl", "ppo_discrete_cnn_vmat2d.py")

sys.argv = [CLEANRL_SCRIPT] + sys.argv[1:]
runpy.run_path(CLEANRL_SCRIPT, run_name="__main__")
