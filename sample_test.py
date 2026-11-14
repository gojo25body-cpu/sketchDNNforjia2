import os, time, torch
from collections import OrderedDict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config
from dataset import SketchDataset
from diffusion_model import GD3PM

CKPT = r"C:\Users\ylliu\OneDrive\Desktop\SketchDNN\runs\train_try_1\checkpoints\epoch_26.pth"
NUM_SAMPLES = 4
OUT_DIR = "samples"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("裝置:", device)

model = GD3PM(device)
sd = torch.load(CKPT, map_location=device, weights_only=False)["model"]
sd = OrderedDict((k.removeprefix("module."), v) for k, v in sd.items())
model.load_state_dict(sd, strict=True)
model.eval()
print("權重載入成功")

torch.manual_seed(0)
t0 = time.time()
nodes = model.sample(NUM_SAMPLES)
print(f"取樣完成，耗時 {time.time() - t0:.1f} 秒")

os.makedirs(OUT_DIR, exist_ok=True)
torch.save(nodes.cpu(), os.path.join(OUT_DIR, "samples.pt"))
for i in range(NUM_SAMPLES):
    fig = SketchDataset.render_graph(nodes[i].cpu())
    fig.savefig(os.path.join(OUT_DIR, f"sample_{i}.png"), dpi=150)
    plt.close(fig)
print("圖片已存到", OUT_DIR)