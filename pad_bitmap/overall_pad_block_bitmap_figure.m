% Load the data
load('bitmap_collection.mat');

% Construct PAD_BITMAP
PAD_BITMAP = zeros(size(CRITICAL_PAD_BITMAP));
PAD_BITMAP(CRITICAL_PAD_BITMAP == 1) = 1;
PAD_BITMAP(REDUNDANT_PAD_BITMAP == 1) = 2;
PAD_BITMAP(DUMMY_PAD_BITMAP == 1) = 3;

% Define colormap
cmap = [
    1.0, 0.5, 0.5;   % Red - Critical
    0.4, 0.4, 0.9;   % Blue - Redundant
    0.8, 0.8, 0.8    % Gray - Dummy
];

% Create figure
figure('Position', [100, 100, 700, 730]);
imagesc(PAD_BITMAP);
colormap(cmap);
axis equal;
axis tight;
set(gca, 'YDir', 'normal');

% Tick and grid styling
ax = gca;
ax.FontName = 'Times New Roman';     % Serif font
ax.FontSize = 24;                    % Larger font size
ax.LineWidth = 1;                    % Thicker axis lines
ax.TickDir = 'out';                  % Ticks pointing out
ax.Box = 'off';                      % No top/right box
% Set tick marks every 200 units
yticks(0:200:size(PAD_BITMAP, 1));
xticks(0:200:size(PAD_BITMAP, 2));

% Multiply tick labels by 10
xticklabels(arrayfun(@(x) num2str(x * 10), xticks(), 'UniformOutput', false));
yticklabels(arrayfun(@(y) num2str(y * 10), yticks(), 'UniformOutput', false));

% Labels with LaTeX interpreter
xlabel('X Position ($\mu m$)', 'Interpreter', 'latex', 'FontSize', 28);
ylabel('Y Position ($\mu m$)', 'Interpreter', 'latex', 'FontSize', 28);
% title('Pad Block Bitmap', 'Interpreter', 'latex', 'FontSize', 28);

% Add legend using dummy patches
hold on;
red_patch  = patch(NaN, NaN, [1.0, 0.7, 0.7]);
blue_patch = patch(NaN, NaN, [0.7, 0.7, 1.0]);
gray_patch = patch(NaN, NaN, [0.8, 0.8, 0.8]);
% lgd = legend([red_patch, blue_patch, gray_patch], ...
%     {'\textbf{Critical Pads}', '\textbf{Redundant Pads}', '\textbf{Dummy Pads}'}, ...
%     'Interpreter', 'latex', ...
%     'Location', 'southoutside', ...
%     'Orientation', 'horizontal', ...
%     'FontSize', 16);
% lgd.Position(1) = lgd.Position(1) + 0.06;  % Move right (increase X)
% lgd.Position(2) = lgd.Position(2) - 0.08;  % Move right (increase X)
hold off
axis off
exportgraphics(gcf, 'overall_pad_bitmap.png', 'Resolution', 600);
