function [scores, embeddings] = yamnetPredict(net, audio)
    % Define YAMNet default parameters
    sampleRate = 16000;

    % Validate inputs
    if nargin < 2
        error('Not enough input arguments. Expected: net, audio.');
    end
    if isempty(audio)
        warning('Empty audio input. Returning empty outputs.');
        scores = [];
        embeddings = [];
        return;
    end

    % Ensure mono
    fprintf('Input audio size: [%d x %d]\n', size(audio));
    if size(audio, 2) > 1
        audio = mean(audio, 2);
    end

    % Resample to 16kHz
    fprintf('Audio size before resampling: [%d x %d]\n', size(audio));
    if round(size(audio, 1) / sampleRate) ~= 4 % Ensure ~4 seconds
        audio = resample(audio, sampleRate, 16000);
    end
    fprintf('Audio size after resampling: [%d x %d]\n', size(audio));

    % Normalize amplitude
    maxAmp = max(abs(audio));
    if maxAmp < 0.1
        audio = audio * (1 / max(0.1, maxAmp));
        fprintf('Amplified audio: MaxAmp=%f\n', max(abs(audio)));
    end

    % Frame audio into patches
    patchWindow = round(0.96 * sampleRate); % 0.96 seconds
    patchHop = round(0.48 * sampleRate); % 0.48 seconds
    frames = buffer(audio, patchWindow, patchWindow - patchHop, 'nodelay')';
    fprintf('Frames size: [%d x %d]\n', size(frames));

    % Extract log-mel spectrogram for each frame
    numFrames = size(frames, 1);
    melBands = 64;  % YAMNet expects 64 mel bands
    timeFrames = 96; % YAMNet expects 96 time frames
    features = zeros(timeFrames, melBands, numFrames); % Preallocate [time x mel x frame]

    for i = 1:numFrames
        s = extractLogMel(frames(i, :), sampleRate, melBands);  % [time x mel]
        fprintf('Mel spectrogram %d size: [%d x %d]\n', i, size(s));
        if isempty(s) || any(isnan(s(:))) || any(isinf(s(:)))
            warning('Invalid mel spectrogram for frame %d. Skipping.', i);
            continue;
        end
        % Resize to 96 time frames
        if size(s, 1) ~= timeFrames
            s = imresize(s, [timeFrames, melBands], 'nearest');
        end
        features(:, :, i) = s;  % [time x mel x frame]
    end

    % Trim invalid frames
    validFrames = find(any(features ~= 0, [1 2]), 1, 'last');
    if isempty(validFrames)
        warning('❌ No valid frames extracted. Returning empty embeddings.');
        scores = [];
        embeddings = [];
        return;
    end
    features = features(:, :, 1:validFrames);
    fprintf('Features size before normalization: [%d x %d x %d]\n', size(features));

    % Normalize features element-wise
    maxVal = max(abs(features(:)));
    if maxVal < 1e-6
        maxVal = 1e-6;
    end
    features = features ./ maxVal; % Element-wise division
    fprintf('Features size after normalization: [%d x %d x %d], max=%f\n', size(features), max(abs(features(:))));

    % Reshape for YAMNet: [time x mel x channels x batch]
    features = permute(features, [1 2 3]); % [time x mel x frame]
    features = reshape(features, [timeFrames, melBands, 1, validFrames]); % [96 x 64 x 1 x N]
    fprintf('Features size for YAMNet: [%d x %d x %d x %d]\n', size(features));

    % Wrap as dlarray
    dlIn = dlarray(features, 'SSCB'); % Spatial x Spatial x Channels x Batch
    fprintf('dlarray size: [%d x %d x %d x %d]\n', size(dlIn));

    % Run prediction
    try
        scores = predict(net, dlIn); % Classification output [521 x N]
        embeddings = forward(net, dlIn, 'Outputs', 'conv2d_12'); % Expected [1 x D x 1024 x N]
        % Log raw embeddings size
        sz = size(embeddings);
        fprintf('Raw embeddings size: [%s]\n', num2str(sz));
        % Squeeze singleton dimensions
        embeddings = squeeze(embeddings);
        sz = size(embeddings);
        fprintf('Squeezed embeddings size: [%s]\n', num2str(sz));
        % Reshape to [1024 x N]
        if numel(sz) >= 3
            % Handle [D1 x D2 x 1024 x N] or similar
            if sz(3) == 1024 && sz(4) == validFrames
                % Case: [D1 x D2 x 1024 x N]
                embeddings = reshape(embeddings, [], 1024, validFrames); % [D1*D2 x 1024 x N]
                embeddings = squeeze(embeddings(1, :, :)); % [1024 x N]
            elseif any(sz == 1024)
                % Find 1024 dimension and permute to [1024 x N]
                idx1024 = find(sz == 1024, 1);
                idxN = find(sz == validFrames, 1);
                if isempty(idxN)
                    idxN = find(sz > 1, 1, 'last'); % Approximate N
                end
                permOrder = [idx1024, idxN, setdiff(1:numel(sz), [idx1024 idxN])];
                embeddings = permute(embeddings, permOrder);
                embeddings = reshape(embeddings, 1024, validFrames);
            else
                warning('Unexpected embedding dimensions: [%s]. Attempting reshape to [1024 x %d].', num2str(sz), validFrames);
                embeddings = reshape(embeddings, 1024, validFrames);
            end
        elseif size(embeddings, 1) ~= 1024
            warning('Unexpected embedding size: [%s]. Reshaping to [1024 x %d].', num2str(sz), validFrames);
            embeddings = reshape(embeddings, 1024, validFrames);
        end
        fprintf('Final embeddings size: [%d x %d]\n', size(embeddings));
        fprintf('Scores size: [%d x %d]\n', size(scores));
    catch e
        warning('Prediction failed: %s. Returning empty embeddings.', e.message);
        scores = [];
        embeddings = [];
    end
end

function mel = extractLogMel(audio, fs, numBands)
    win = hamming(400, 'periodic');
    overlap = 240;
    fftLength = 512;
    [~, ~, ~, ps] = spectrogram(audio, win, overlap, fftLength, fs);
    fprintf('Power spectrogram size: [%d x %d]\n', size(ps));
    
    ps = abs(ps).^2;
    if all(ps(:) == 0)
        warning('Zero power spectrogram for audio. Adding noise.');
        ps = ps + 1e-10;
    end
    
    % Design mel filterbank
    try
        fb = designAuditoryFilterBank(fs, ...
            'FFTLength', fftLength, ...
            'NumBands', numBands, ...
            'FrequencyRange', [125, 7500], ...
            'Normalization', 'bandwidth');
        fprintf('Filterbank size: [%d x %d]\n', size(fb));
    catch e
        warning('Failed to design mel filterbank: %s. Returning empty spectrogram.', e.message);
        mel = [];
        return;
    end
    
    mel = fb * ps;  % [mel x time]
    fprintf('Mel spectrogram size before transpose: [%d x %d]\n', size(mel));
    mel = log(mel + 1e-6);
    if any(isnan(mel(:))) || any(isinf(mel(:)))
        warning('Invalid mel spectrogram: NaN=%d, Inf=%d. Using zeros.', any(isnan(mel(:))), any(isinf(mel(:))));
        mel = zeros(size(mel));
    end
    mel = mel'; % [time x mel]
    fprintf('Mel spectrogram size after transpose: [%d x %d]\n', size(mel));
end