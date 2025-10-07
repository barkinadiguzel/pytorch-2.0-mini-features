import torch
import torch.nn as nn
import time


class SimpleModel(nn.Module):
def __init__(self):
super().__init__()
self.fc = nn.Linear(128, 64)


def forward(self, x):
return torch.relu(self.fc(x))


model = SimpleModel()
compiled_model = torch.compile(model)


x = torch.randn(1, 128)


# Measure runtime
t0 = time.time()
for _ in range(100):
_ = model(x)
t1 = time.time()


base_time = (t1 - t0) / 100


# Compiled
x = torch.randn(1, 128)
t0 = time.time()
for _ in range(100):
_ = compiled_model(x)
t1 = time.time()


compiled_time = (t1 - t0) / 100


print(f"Base model avg time: {base_time*1000:.3f} ms")
print(f"Compiled model avg time: {compiled_time*1000:.3f} ms")
print(f"Speedup: {base_time / compiled_time:.2f}x")
