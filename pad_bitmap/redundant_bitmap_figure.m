% % Load data
% load('REDUNDANT_MAIN_PAD_BLOCK_BITMAP.mat');  % variable: REDUNDANT_MAIN_PAD_BLOCK_BITMAP
% 
% % Create binary mask
% MAIN_PAD_BITMAP = zeros(size(REDUNDANT_MAIN_PAD_BLOCK_BITMAP));
% MAIN_PAD_BITMAP(REDUNDANT_MAIN_PAD_BLOCK_BITMAP == 1) = 1;
% 
% % Define colors
% foreground = [0.4, 0.4, 0.9];   % 
% background = [0.8, 0.8, 0.8];   % 
% 
% % Construct RGB image
% RGB = zeros([size(MAIN_PAD_BITMAP), 3]);
% for i = 1:3
%     RGB(:, :, i) = background(i) + (foreground(i) - background(i)) * MAIN_PAD_BITMAP;
% end
% 
% % Plot
% figure('Position', [100, 100, 700, 700]);
% imshow(RGB, 'InitialMagnification', 'fit');  % 或 'nearest'
% 
% axis off;
% 
% % Export with high resolution
% exportgraphics(gcf, 'redundant_main_pad_bitmap.png', 'Resolution', 600);
% 
% 
% % Load data
% load('REDUNDANT_MAIN_PAD_BLOCK_BITMAP_DILATED.mat');  % variable: REDUNDANT_MAIN_PAD_BLOCK_BITMAP_DILATED
% 
% % Create binary mask
% MAIN_PAD_BITMAP_DILATED = zeros(size(REDUNDANT_MAIN_PAD_BLOCK_BITMAP_DILATED));
% MAIN_PAD_BITMAP_DILATED(REDUNDANT_MAIN_PAD_BLOCK_BITMAP_DILATED == 1) = 1;
% 
% % Define colors
% foreground = [0.4, 0.4, 0.9];   % 
% background = [0.8, 0.8, 0.8];   % 
% 
% % Construct RGB image
% RGB = zeros([size(MAIN_PAD_BITMAP_DILATED), 3]);
% for i = 1:3
%     RGB(:, :, i) = background(i) + (foreground(i) - background(i)) * MAIN_PAD_BITMAP_DILATED;
% end
% 
% % Plot
% figure('Position', [100, 100, 700, 700]);
% imshow(RGB, 'InitialMagnification', 'fit');  % 或 'nearest'
% 
% axis off;
% 
% % Export with high resolution
% exportgraphics(gcf, 'redundant_main_pad_bitmap_dilated.png', 'Resolution', 600);
% 
% 
% 
% % Load data
% load('REDUNDANT_COPY_PAD_BLOCK_BITMAP.mat');  % variable: REDUNDANT_COPY_PAD_BLOCK_BITMAP
% 
% % Create binary mask
% COPY_PAD_BITMAP = zeros(size(REDUNDANT_COPY_PAD_BLOCK_BITMAP));
% COPY_PAD_BITMAP(REDUNDANT_COPY_PAD_BLOCK_BITMAP == 1) = 1;
% 
% % Define colors
% foreground = [0.4, 0.4, 0.9];   % 
% background = [0.8, 0.8, 0.8];   % 
% 
% % Construct RGB image
% RGB = zeros([size(COPY_PAD_BITMAP), 3]);
% for i = 1:3
%     RGB(:, :, i) = background(i) + (foreground(i) - background(i)) * COPY_PAD_BITMAP;
% end
% 
% % Plot
% figure('Position', [100, 100, 700, 700]);
% imshow(RGB, 'InitialMagnification', 'fit');  % 或 'nearest'
% 
% axis off;
% 
% % Export with high resolution
% exportgraphics(gcf, 'redundant_copy_pad_bitmap.png', 'Resolution', 600);
% 
% 
% % Load data
% load('REDUNDANT_COPY_PAD_BLOCK_BITMAP_DILATED.mat');  % variable: REDUNDANT_COPY_PAD_BLOCK_BITMAP_DILATED
% 
% % Create binary mask
% COPY_PAD_BITMAP_DILATED = zeros(size(REDUNDANT_COPY_PAD_BLOCK_BITMAP_DILATED));
% COPY_PAD_BITMAP_DILATED(REDUNDANT_COPY_PAD_BLOCK_BITMAP_DILATED == 1) = 1;
% 
% % Define colors
% foreground = [0.4, 0.4, 0.9];   % 
% background = [0.8, 0.8, 0.8];   % 
% 
% % Construct RGB image
% RGB = zeros([size(COPY_PAD_BITMAP_DILATED), 3]);
% for i = 1:3
%     RGB(:, :, i) = background(i) + (foreground(i) - background(i)) * COPY_PAD_BITMAP_DILATED;
% end
% 
% % Plot
% figure('Position', [100, 100, 700, 700]);
% imshow(RGB, 'InitialMagnification', 'fit');  % 或 'nearest'
% 
% axis off;
% 
% % Export with high resolution
% exportgraphics(gcf, 'redundant_copy_pad_bitmap_dilated.png', 'Resolution', 600);
% 
% 
% 
% % AND BITMAP of MAIN & COPY
% % Load data
% load('cross_bitmap_main_copy.mat');  % variable: cross_bitmap_main_copy
% 
% % Create binary mask
% cross_bitmap_main_copy_bitmap = zeros(size(cross_bitmap_main_copy));
% cross_bitmap_main_copy_bitmap(cross_bitmap_main_copy == 1) = 1;
% 
% % Define colors
% foreground = [0.4, 0.4, 0.9];   % 
% background = [0.8, 0.8, 0.8];   % 
% 
% % Construct RGB image
% RGB = zeros([size(cross_bitmap_main_copy_bitmap), 3]);
% for i = 1:3
%     RGB(:, :, i) = background(i) + (foreground(i) - background(i)) * cross_bitmap_main_copy_bitmap;
% end
% 
% % Plot
% figure('Position', [100, 100, 700, 700]);
% imshow(RGB, 'InitialMagnification', 'fit');  % 或 'nearest'
% 
% axis off;
% 
% % Export with high resolution
% exportgraphics(gcf, 'redundant_main_and_copy_bitmap_dilated.png', 'Resolution', 600);



% AND BITMAP of MAIN & COPY
% Load data
load('TOTAL_CRITICAL_AREA_BITMAP.mat');  % variable: TOTAL_CRITICAL_AREA_BITMAP

% Create binary mask
TOTAL_CRITICAL_AREA = zeros(size(TOTAL_CRITICAL_AREA_BITMAP));
TOTAL_CRITICAL_AREA(TOTAL_CRITICAL_AREA_BITMAP == 1) = 1;

% Define colors
foreground = [1.0, 0.5, 0.5];   % 
background = [0.8, 0.8, 0.8];   % 

% Construct RGB image
RGB = zeros([size(TOTAL_CRITICAL_AREA), 3]);
for i = 1:3
    RGB(:, :, i) = background(i) + (foreground(i) - background(i)) * TOTAL_CRITICAL_AREA;
end

% Plot
figure('Position', [100, 100, 700, 700]);
imshow(RGB, 'InitialMagnification', 'fit');  % 或 'nearest'

axis off;

% Export with high resolution
exportgraphics(gcf, 'total_critical_area_bitmap.png', 'Resolution', 600);