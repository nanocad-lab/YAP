% Load data
load('CRITICAL_PAD_BITMAP.mat');  % variable: CRITICAL_PAD_BITMAP
% load('CRITICAL_PAD_BLOCK_BITMAP.mat');  % variable: CRITICAL_PAD_BLOCK_BITMAP

% Create binary mask
PAD_BITMAP = zeros(size(CRITICAL_PAD_BITMAP));
PAD_BITMAP(CRITICAL_PAD_BITMAP == 1) = 1;

% Define colors
foreground = [1.0, 0.5, 0.5];   % 
background = [0.8, 0.8, 0.8];   % 

% Construct RGB image
RGB = zeros([size(PAD_BITMAP), 3]);
for i = 1:3
    RGB(:, :, i) = background(i) + (foreground(i) - background(i)) * PAD_BITMAP;
end

% Plot
figure('Position', [100, 100, 700, 700]);
imshow(RGB, 'InitialMagnification', 'fit');  % 或 'nearest'

axis off;

% Export with high resolution
exportgraphics(gcf, 'critical_pad_bitmap.png', 'Resolution', 600);


% Load data
load('CRITICAL_PAD_BITMAP_DILATED.mat');  % variable: CRITICAL_PAD_BITMAP_DILATED

% Create binary mask
PAD_BITMAP_DILATED = zeros(size(CRITICAL_PAD_BITMAP_DILATED));
PAD_BITMAP_DILATED(CRITICAL_PAD_BITMAP_DILATED == 1) = 1;

% Define colors
foreground = [1.0, 0.5, 0.5];   % 
background = [1, 1, 1];   % 

% Construct RGB image
RGB = zeros([size(PAD_BITMAP_DILATED), 3]);
for i = 1:3
    RGB(:, :, i) = background(i) + (foreground(i) - background(i)) * PAD_BITMAP_DILATED;
end

% Plot
figure('Position', [100, 100, 700, 700]);
imshow(RGB, 'InitialMagnification', 'fit');  % 或 'nearest'

% Export with high resolution
exportgraphics(gcf, 'critical_pad_bitmap_dilated.png', 'Resolution', 600);

axis off;

% % Create binary mask
% PAD_BITMAP = zeros(size(CRITICAL_PAD_BLOCK_BITMAP));
% PAD_BITMAP(CRITICAL_PAD_BLOCK_BITMAP == 1) = 1;
% 
% % Define colors
% foreground = [1.0, 0.5, 0.5];   % 
% background = [0.8, 0.8, 0.8];   % 
% 
% % Construct RGB image
% RGB = zeros([size(PAD_BITMAP), 3]);
% for i = 1:3
%     RGB(:, :, i) = background(i) + (foreground(i) - background(i)) * PAD_BITMAP;
% end
% 
% % Plot
% figure('Position', [100, 100, 700, 700]);
% imshow(RGB, 'InitialMagnification', 'fit');  % 或 'nearest'
% 
% % Export with high resolution
% % exportgraphics(gcf, 'critical_pad_bitmap_dilated.png', 'Resolution', 600);
% exportgraphics(gcf, 'critical_pad_bitmap.png', 'Resolution', 600);
