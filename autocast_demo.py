import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import time


device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)


class BigModel(nn.Module):
def __init__(self):
super().__init__()
self.fc1 = nn.Linear(1024, 1024)
self.relu = nn.ReLU()
self.fc2 = nn.Linear(1024, 1024)


def forward(self, x):
return self.fc2(self.relu(self.fc1(x)))


model = BigModel().eval()
x = torch.randn(16, 1024)


# Normal precision
t0 = time.time()
with torch.no_grad():
for _ in range(50):
_ = model(x)
t1 = time.time()
normal_time = (t1 - t0) / 50


# Mixed precision
t0 = time.time()
with torch.no_grad():
with autocast():
for _ in range(50):
_ = model(x)
t1 = time.time()
autocast_time = (t1 - t0) / 50


print(f"Device: {device}")
print(f"Normal precision avg: {normal_time*1000:.3f} ms")
print(f"Mixed precision avg: {autocast_time*1000:.3f} ms")
if autocast_time > 0:
print(f"Speedup: {normal_time / autocast_time:.2f}x")
