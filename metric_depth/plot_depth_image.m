% rgbImage = imread("frame_100_metric_depth.png");
% imshow(rgbImage)

matrix = readmatrix('depth_matrix.csv');
[X,Y]=meshgrid(1:2448,1:2048);
% Plot grid
figure;
h=imagesc(X(:),Y(:),matrix);
colorbar
