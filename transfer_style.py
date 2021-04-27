import torch
import matplotlib.pyplot as plt

from model import device
from utils import image_loader, imshow, run_style_transfer

plt.ion()

#%%

style_img = image_loader("./data/example_images/matrix.jpg")
content_img = image_loader("./data/example_images/janelle.jpg")

#%%

imshow(style_img, title='Style Image')
imshow(content_img, title='Content Image')

#%%

input_img = content_img.clone()
# if you want to use white noise instead uncomment the below line:
# input_img = torch.randn(content_img.data.size()).to(device, torch.float)

# add the original input image to the figure:
imshow(input_img, title='Input Image')

#%%

output = run_style_transfer(content_img, style_img, input_img, num_steps=300, style_weight=1000000, content_weight=1)
imshow(output, title='Output Image')

#%%

plt.ioff()
plt.show()
