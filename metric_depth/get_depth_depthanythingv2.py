import cv2
import torch
import numpy as np

from depth_anything_v2.dpt import DepthAnythingV2

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
}

encoder = 'vitl' # or 'vits', 'vitb'
dataset = 'vkitti' # 'hypersim' for indoor model, 'vkitti' for outdoor model
max_depth = 20 # 20 for indoor model, 80 for outdoor model

model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth', map_location='cpu'))
model.eval()

raw_img = cv2.imread('C:/Users/BAO/Desktop/frame_100.jpg')
depth = model.infer_image(raw_img) # HxW depth map in meters in numpy
# save the matrix to a file
np.savetxt('C:/Users/BAO/Desktop/depth_matrix.csv', depth, delimiter=',')
cv2.imwrite('C:/Users/BAO/Desktop/frame_100_metric_depth.png', depth)