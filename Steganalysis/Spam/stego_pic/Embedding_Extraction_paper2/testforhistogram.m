function zero_crossings=testforhistogram(img_gray)
% Read the image
%img_gray = imread('img/lena.jpg');

% Define parameters
sigma = 1.4; % Standard deviation of the Gaussian kernel
kernel_size = 5; % Size of the kernel (odd number)
threshold = 0.01; % Threshold for edge detection

% Create the Laplacian of Gaussian filter
filter_size = 2 * ceil(3 * sigma) + 1;
log_filter = fspecial('log', filter_size, sigma);

% Apply the filter to the image
filtered_img = imfilter(double(img_gray), log_filter, 'same', 'replicate');

% Find zero-crossings
[rows, cols] = size(filtered_img);
zero_crossings = zeros(rows, cols);

for i = 2:rows-1
    for j = 2:cols-1
        neighbors = [filtered_img(i-1, j), filtered_img(i+1, j), ...
                     filtered_img(i, j-1), filtered_img(i, j+1), ...
                     filtered_img(i-1, j-1), filtered_img(i-1, j+1), ...
                     filtered_img(i+1, j-1), filtered_img(i+1, j+1)];
        max_neighbor = max(neighbors);
        min_neighbor = min(neighbors);
        
        % If the product of the neighbors is negative, it indicates a zero-crossing
        if max_neighbor * min_neighbor < 0 && abs(max_neighbor - min_neighbor) > threshold
            zero_crossings(i, j) = 1;
        end
    end
end

% Display the edge image
imshow(zero_crossings);
title('Edges using Laplacian of Gaussian');
end