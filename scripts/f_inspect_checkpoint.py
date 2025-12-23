import argparse
import torch
import os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()

    path = args.ckpt
    print("ckpt:", path)
    print("exists:", os.path.exists(path))

    obj = torch.load(path, map_location="cpu")
    print("type:", type(obj))

    if isinstance(obj, dict):
        print("dict keys:", list(obj.keys()))
        for k in list(obj.keys())[:20]:
            v = obj[k]
            print(f"  {k}: {type(v)}")
        # if it looks like a state_dict, show one tensor
        tensor_keys = [k for k,v in obj.items() if hasattr(v, "shape")]
        if tensor_keys:
            tk = tensor_keys[0]
            print("example tensor key:", tk, "shape:", obj[tk].shape)
    else:
        # maybe it's a whole nn.Module or Agent
        if hasattr(obj, "state_dict"):
            sd = obj.state_dict()
            print("has state_dict, keys:", list(sd.keys())[:20])
        else:
            print("no state_dict attribute")

if __name__ == "__main__":
    main()
