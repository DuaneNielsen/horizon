# importing image object from PIL
import math
from PIL import Image, ImageDraw
from torchvision.io.image import read_image
from torchvision.transforms.functional import to_pil_image
from dataset import HorizonDataSet, reverse_norm

w, h = 220, 190
shape = [(40, 40), (w - 10, h - 10)]

# creating new Image object
ds = HorizonDataSet(data_dir='../data/horizon', image_size=64, rotate=False, select_label='complex_mean')

img, complex_mean = ds[6]
img = reverse_norm(img).byte()
img = to_pil_image(img)

# create line image
img1 = ImageDraw.Draw(img)
img1.line([(32, 32), (complex_mean.real * 32 + 32, complex_mean.imag * 32 + 32)], fill="red", width=0)
img.show()