% Load data
load('line_defect.mat');  % variable: line_defect

% Create binary mask
PAD_BITMAP = zeros(size(line_defect));
PAD_BITMAP(line_defect == 1) = 1;

% Define colors
foreground = [255, 199, 44]/255;   % orange
background = [0, 31, 98]/255;   % dark blue

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
exportgraphics(gcf, 'line_defect.png', 'Resolution', 300);
