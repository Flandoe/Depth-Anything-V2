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
max_depth = 80 # 20 for indoor model, 80 for outdoor model

model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth', map_location='cpu'))
model.eval()

# TODO: intrinsic parameters
cameraMatrix = np.array([[1443.65026, 0, 1163.27334],
                     [0, 1444.14432, 1044.59984],
                     [0, 0, 1]])
distCoeffs = np.array([0, 0, 0, 0, 0])

detection_filename = "D:\\OneDrive - The University Of Hong Kong\\SPP\\RnD\\2\\2024-03-15-12-47-01.txt" # TODO
image_directory = "D:\\MiC\\images\\2\\" # TODO
frame_id_old = -1
depth_old = []
output_filename = detection_filename.split(".")[0] + "_position.txt"
# output_filename = "C:\\Users\\BAO\\Desktop\\_position.txt"
with open(detection_filename, 'r') as file, open(output_filename, "w") as output_file:
    tmp_list = ['x','y','z','videoname','frameid','objid','left','top','bbox_w','bbox_h','a1','a2','a3','a4','objname','b1']
    output_file.write('\t'.join(tmp_list))
    output_file.write('\n')
    data = file.readlines()
    for line in data:
        elements = line.split(' ')
        frame_id = int(elements[1]) # image frames 
        pixel_x = float(elements[3]) + float(elements[5])/2.0 # bbox_left, bbox_top, bbox_w, bbox_h
        pixel_y = float(elements[4]) + float(elements[6])/2.0
        if len(elements) == 14:
            elements[-3:-1] = ['_'.join(map(str, elements[-3:-1]))]
        elif len(elements) == 15:
            elements[-4:-1] = ['_'.join(map(str, elements[-4:-1]))]
        print(frame_id,pixel_x,pixel_y)
        if frame_id == frame_id_old:
            depth = depth_old
            print("same " + str(frame_id))
        else:
            raw_img = cv2.imread(image_directory+'frame_{}.jpg'.format(frame_id))
            depth = model.infer_image(raw_img) # HxW depth map in meters in numpy  
            depth_old = depth # update depth  
            frame_id_old = frame_id # update frame_id
        Zc = depth[int(pixel_y), int(pixel_x)]
        # projection process
        pixel = np.array([pixel_x, pixel_y], dtype='float32').reshape(1,1,2)
        undistorted_pixel = cv2.undistortPoints(pixel, cameraMatrix, distCoeffs, P=cameraMatrix)
        point2D = undistorted_pixel.reshape(-1, 2)
        point2D_op = np.hstack((point2D, np.ones((1, 1))))
        uvPoint = point2D_op.reshape(3, 1)
        kMat_inv = np.linalg.inv(cameraMatrix)
        temp1 = Zc*uvPoint
        camera_3d = np.matmul(kMat_inv, temp1)
        # print(camera_3d)
        # line_new = "{:.2f}".format(camera_3d[0][0]) + " " + "{:.2f}".format(camera_3d[1][0]) + " " + "{:.2f}".format(camera_3d[2][0]) + " " + line.strip() +"\n"
        # output_file.write('\t'.join(line_new))
        line_new = "{:.2f}".format(camera_3d[0][0]) + "\t" + "{:.2f}".format(camera_3d[1][0]) + "\t" + "{:.2f}".format(camera_3d[2][0]) + "\t" + '\t'.join(elements) +"\n"
        # print(line_new)
        output_file.write(line_new)
        output_file.flush()
    exit()
