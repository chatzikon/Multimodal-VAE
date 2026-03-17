from PIL import Image
from torchvision import transforms


path='/home/chatziko/PycharmProjects/PythonProject/cross_modal_vae/pose_data/B1Counting/BB_left_0.png'

img=Image.open(path).convert('RGB')
transform = transforms.ToTensor()

img_t = transform(img)
xx=1