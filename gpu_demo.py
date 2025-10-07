import torch
import torch.nn as nn


# Set the default device globally
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)


class SimpleNet(nn.Module):
def __init__(self):
super().__init__()
self.fc1 = nn.Linear(8, 16)
self.relu = nn.ReLU()
self.fc2 = nn.Linear(16, 4)


def forward(self, x):
return self.fc2(self.relu(self.fc1(x)))


model = SimpleNet()
x = torch.randn(2, 8)
y = model(x)


print(f"Global device: {device}")
print(f"Input device: {x.device}")
print(f"Output device: {y.device}")
