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

# TODO: intrinsic parameters
cameraMatrix = np.array([[187.58, 0, 360],
                     [0, 200.08, 288],
                     [0, 0, 1]])
distCoeffs = np.array([0, 0, 0, 0, 0])
img_size = [720,576] # width,height

detection_filename = "D:\\PedestrianSpeed\\prediction\\1 20210630_0755-0845 SL6335_ch1.txt" # TODO
image_directory = "D:\\PedestrianSpeed\\video\\Rt.1 20210630_0755-0845 SL6335_ch1\\" # TODO
frame_id_old = -1
depth_old = []
output_filename = detection_filename.split(".")[0] + "_position.txt"
# output_filename = "C:\\Users\\BAO\\Desktop\\_position.txt"
with open(detection_filename, 'r') as file, open(output_filename, "w") as output_file:
    tmp_list = ['x','y','z','frameid','objid','left','top','bbox_w','bbox_h','score','7','8','9']
    output_file.write('\t'.join(tmp_list))
    output_file.write('\n')
    data = file.readlines()
    for line in data:
        elements = line.split(',')
        frame_id = int(elements[0]) # image frames 
        pixel_x = float(elements[2]) + float(elements[4])/2.0 # bbox_left, bbox_top, bbox_w, bbox_h
        pixel_y = float(elements[3]) + float(elements[5])/2.0
        print(frame_id,pixel_x,pixel_y, end="\t")
        if int(pixel_x) > img_size[0] or int(pixel_y) > img_size[1]:
            print("\033[91m frame: {} objid: {} detection error! \033[0m".format(frame_id, int(elements[1])))
            continue
        if frame_id == frame_id_old:
            depth = depth_old
            # print("same " + str(frame_id), end="\t", flush=True)
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
        line_new = "{:.2f}".format(camera_3d[0][0]) + "\t" + "{:.2f}".format(camera_3d[1][0]) + "\t" + "{:.2f}".format(camera_3d[2][0]) + "\t" + '\t'.join(elements)
        # print(line_new)
        output_file.write(line_new)
        output_file.flush()
    exit()
