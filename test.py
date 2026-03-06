import torch
from torchvision.models import vgg16, VGG16_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
model = model.features[:16]
model = model.to(device)

print("Success")