"""
Setup verification script for Accessible RetNet project.
Run this first to confirm CUDA, GPU, and all dependencies are ready.
"""
import sys

def check_python():
    v = sys.version_info
    status = "OK" if v.major == 3 and v.minor >= 9 else "WARN"
    print(f"  [Python {status}] {sys.version.split()[0]}")
    return status == "OK"

def check_torch():
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        if cuda_ok:
            dev = torch.cuda.get_device_name(0)
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)) / 1024**3
            cc = torch.cuda.get_device_capability(0)
            print(f"  [PyTorch OK] {torch.__version__}")
            print(f"  [CUDA    OK] Device: {dev}")
            print(f"  [VRAM    OK] Total: {total:.1f}GB  |  Free: {free:.1f}GB")
            print(f"  [Compute CC] {cc[0]}.{cc[1]} (need >=7.0 for FP16 Tensor Cores)")
            if cc[0] >= 7:
                print(f"  [FP16    OK] Tensor Core acceleration available")
            else:
                print(f"  [FP16  WARN] No Tensor Cores — FP16 will still work but slower")
        else:
            print(f"  [PyTorch OK] {torch.__version__}")
            print(f"  [CUDA  FAIL] CUDA not available! Check NVIDIA drivers.")
            print(f"             Run: pip install torch==2.1.2+cu118 --index-url https://download.pytorch.org/whl/cu118")
        return cuda_ok
    except ImportError:
        print(f"  [PyTorch FAIL] Not installed. Run: pip install -r requirements.txt")
        return False

def check_packages():
    packages = {
        "transformers": "4.38.0",
        "datasets": "2.18.0",
        "tokenizers": "0.15.0",
        "numpy": "1.26.0",
        "tqdm": "4.66.0",
        "rich": "13.7.0",
    }
    all_ok = True
    for pkg, min_ver in packages.items():
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "?")
            print(f"  [{pkg:15s} OK] v{ver}")
        except ImportError:
            print(f"  [{pkg:15s} FAIL] Not installed")
            all_ok = False
    return all_ok

def quick_forward_test():
    """Run a tiny forward pass to confirm GPU compute works."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        device = torch.device("cuda")
        x = torch.randn(2, 16, 768, device=device, dtype=torch.float16)
        w = torch.randn(768, 768, device=device, dtype=torch.float16)
        y = x @ w
        torch.cuda.synchronize()
        print(f"  [GPU Test OK] FP16 matmul on CUDA succeeded. Output shape: {tuple(y.shape)}")
        return True
    except Exception as e:
        print(f"  [GPU Test FAIL] {e}")
        return False

def main():
    print("\n" + "="*60)
    print("  Accessible RetNet — Environment Check")
    print("="*60)

    print("\n[1/4] Python Version")
    py_ok = check_python()

    print("\n[2/4] PyTorch + CUDA")
    cuda_ok = check_torch()

    print("\n[3/4] Package Dependencies")
    pkg_ok = check_packages()

    print("\n[4/4] GPU Compute Test (FP16)")
    gpu_test_ok = quick_forward_test() if cuda_ok else False

    print("\n" + "="*60)
    if py_ok and cuda_ok and pkg_ok and gpu_test_ok:
        print("  RESULT: ALL CHECKS PASSED — Ready to train!")
    else:
        print("  RESULT: SOME CHECKS FAILED — Fix above issues before training.")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
