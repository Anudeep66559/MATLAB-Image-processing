clc
clear
close all

%% ======== VIDEO INPUT ========
videoPath = 'C:\Users\ce23b\OneDrive\Desktop\sediments project\45lps\12mm 45lps\C0327.MP4';
v = VideoReader(videoPath);

%% ======== BUILD BACKGROUND MODEL ========
disp('Building background model...')
nBgFrames = min(30, floor(v.Duration * v.FrameRate));

% Read first frame to get dimensions
firstFrame = readFrame(v);
gFirst     = im2double(rgb2gray(firstFrame));
illum      = imboxfilt(gFirst, 101);
corr       = mat2gray(gFirst ./ (illum + 0.01));
[H, W]     = size(corr);

% Pre-allocate with single precision — uses half the RAM of double
bgStack = zeros(H, W, nBgFrames, 'single');
bgStack(:,:,1) = single(corr);

for k = 2:nBgFrames
    if hasFrame(v)
        f     = readFrame(v);
        g     = im2double(rgb2gray(f));
        illum = imboxfilt(g, 101);
        corr  = mat2gray(g ./ (illum + 0.01));
        bgStack(:,:,k) = single(corr);
    end
end

bgModel  = median(bgStack, 3);   % static background
bgStdDev = std(bgStack, 0, 3);   % per-pixel noise
clear bgStack                     % free RAM immediately after use

v = VideoReader(videoPath);       % reset to first frame

%% ======== OUTPUT VIDEO ========
outputVideo = VideoWriter('FINAL_BW_Output.avi', 'Motion JPEG AVI');
outputVideo.FrameRate = v.FrameRate;
open(outputVideo)

%% ======== TRACKING PARAMETERS (tune if needed) ========
MIN_AREA       = 600;
MAX_AREA       = 8000;
MAX_ASPECT     = 3.5;
SEARCH_RADIUS  = 150;
BG_DIFF_THRESH = 0.12;
MAX_MISSED     = 10;

%% ======== KALMAN FILTER SETUP ========
kalman = configureKalmanFilter('ConstantVelocity', ...
    [0 0], [100 25], [25 10], 25);
kalmanInitialized = false;

%% ======== MAIN PROCESSING LOOP ========
data           = [];
frameNo        = 1;
adaptiveThresh = 0.18;
missedFrames   = 0;

disp('Processing video...')
while hasFrame(v)
    frame = readFrame(v);

    %% --- Illumination correction ---
    g_d       = im2double(rgb2gray(frame));
    illum     = imboxfilt(g_d, 101);
    corrected = single(mat2gray(g_d ./ (illum + 0.01)));  % single precision
    smoothImg = imgaussfilt(corrected, 3);

    %% --- Layer 1: Adaptive threshold ---
    otsuThresh     = graythresh(smoothImg) * 0.75;
    adaptiveThresh = 0.95*adaptiveThresh + 0.05*otsuThresh;
    bw_thresh      = smoothImg < adaptiveThresh;

    %% --- Layer 2: Background subtraction ---
    bgDiff  = abs(smoothImg - bgModel);
    bw_diff = bgDiff > (BG_DIFF_THRESH + 2*bgStdDev);
    bw      = bw_thresh & bw_diff;

    %% --- Morphological cleanup ---
    bw = bwareaopen(bw, MIN_AREA);
    bw = imclose(bw, strel('disk', 6));
    bw = imdilate(bw, strel('disk', 3));
    bw = imfill(bw, 'holes');

    %% --- Layer 3: Shape filtering ---
    stats = regionprops(bw, 'Centroid', 'Area', ...
                        'MajorAxisLength', 'MinorAxisLength', 'PixelIdxList');

    validIdx = false(1, numel(stats));
    for i = 1:numel(stats)
        a      = stats(i).Area;
        aspect = stats(i).MajorAxisLength / (stats(i).MinorAxisLength + 0.001);
        if a >= MIN_AREA && a <= MAX_AREA && aspect <= MAX_ASPECT
            validIdx(i) = true;
        end
    end
    stats = stats(validIdx);

    %% --- Layer 4: Kalman-guided blob selection ---
    detectedPos      = [];
    detectedPixelIdx = [];
    if ~isempty(stats)
        centers = reshape([stats.Centroid], 2, [])';
        if kalmanInitialized
            pred  = predict(kalman);
            dists = sqrt(sum((centers - pred).^2, 2));
            [minDist, idx] = min(dists);
            if minDist < SEARCH_RADIUS
                detectedPos      = centers(idx, :);
                detectedPixelIdx = stats(idx).PixelIdxList;
            end
        else
            [~, idx] = max([stats.Area]);
            detectedPos      = centers(idx, :);
            detectedPixelIdx = stats(idx).PixelIdxList;
        end
    end

    %% --- Kalman update ---
    if ~isempty(detectedPos)
        missedFrames = 0;
        if ~kalmanInitialized
            kalman = configureKalmanFilter('ConstantVelocity', ...
                detectedPos, [100 25], [25 10], 25);
            kalmanInitialized = true;
        end
        smoothPos = correct(kalman, detectedPos);
        data = [data; frameNo round(smoothPos(1)) round(smoothPos(2))];
    else
        if kalmanInitialized
            predict(kalman);
        end
        missedFrames = missedFrames + 1;
        if missedFrames > MAX_MISSED
            kalmanInitialized = false;
            missedFrames = 0;
        end
    end

    %% --- Build and write output frame ---
    bw_clean = false(size(bw));
    if ~isempty(detectedPixelIdx)
        bw_clean(detectedPixelIdx) = true;
    end
    frame_out = uint8(~bw_clean) * 255;
    writeVideo(outputVideo, cat(3, frame_out, frame_out, frame_out));

    frameNo = frameNo + 1;
end

close(outputVideo)

%% ======== SAVE COORDINATES ========
if ~isempty(data)
    T = table(data(:,1), data(:,2), data(:,3), ...
        'VariableNames', {'Frame', 'X_pixel', 'Y_pixel'});
    writetable(T, 'FINAL_particle_coordinates.xlsx');
    fprintf('Particle detected in %d frames.\n', size(data,1));
else
    disp('No particle detected.')
end

disp('Done.')

