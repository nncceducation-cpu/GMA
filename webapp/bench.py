"""Is the GPU itself slow, or just our use of it?"""
import sys, time
sys.path.insert(0, "/app")
import torch

print("torch", torch.__version__, "| device", torch.cuda.get_device_name(0))
print("capability", torch.cuda.get_device_capability(0))
print("arch list  ", torch.cuda.get_arch_list())

# --- 1. raw matmul: is the GPU fast at all? -------------------------------
a = torch.randn(4096, 4096, device="cuda")
for _ in range(3):
    a @ a
torch.cuda.synchronize()
t = time.time()
for _ in range(20):
    a @ a
torch.cuda.synchronize()
dt = (time.time() - t) / 20
tflops = 2 * 4096**3 / dt / 1e12
print("\nMATMUL fp32 4096^3 : %.4fs -> %.1f TFLOPS" % (dt, tflops))

with torch.autocast("cuda", dtype=torch.float16):
    for _ in range(3):
        a @ a
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(20):
        a @ a
    torch.cuda.synchronize()
dt16 = (time.time() - t) / 20
print("MATMUL fp16 4096^3 : %.4fs -> %.1f TFLOPS" % (dt16, 2*4096**3/dt16/1e12))

# --- 2. ViTPose backbone: fp32 vs fp16 ------------------------------------
from pipeline.pose_extract import PoseExtractor
p = PoseExtractor(device="cuda")
pose_model, _ = p._load()
x = torch.randn(24, 3, 256, 192, device="cuda")

with torch.no_grad():
    for _ in range(2):
        pose_model.backbone(x)
    torch.cuda.synchronize()
    t = time.time()
    pose_model.backbone(x)
    torch.cuda.synchronize()
    fp32 = time.time() - t

    with torch.autocast("cuda", dtype=torch.float16):
        for _ in range(2):
            pose_model.backbone(x)
        torch.cuda.synchronize()
        t = time.time()
        pose_model.backbone(x)
        torch.cuda.synchronize()
        fp16 = time.time() - t

print("\nVITPOSE-H batch24 fp32 : %6.2fs -> %5.1f fps" % (fp32, 24/fp32))
print("VITPOSE-H batch24 fp16 : %6.2fs -> %5.1f fps  (%.1fx faster)"
      % (fp16, 24/fp16, fp32/fp16))
