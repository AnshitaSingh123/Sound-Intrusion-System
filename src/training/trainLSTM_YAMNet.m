clc; clear;

%% === Load Pre-Extracted YAMNet Features ===
fprintf("📂 Loading YAMNet features...\n");
try
    load('src/feature_extraction/features_yamnet.mat', ...
        'trainFeatures', 'valFeatures', 'testFeatures');
catch e
    error('❌ Failed to load features_yamnet.mat: %s', e.message);
end

% Validate embeddings and labels
fprintf("🔍 Validating embeddings and labels...\n");
numTrain = length(trainFeatures);
numVal = length(valFeatures);
numTest = length(testFeatures);

% Check for valid embeddings and labels
validTrain = find(cellfun(@(x, l) ~isempty(x) && ~any(isnan(x(:))) && ~any(isinf(x(:))) && ...
    ndims(x) == 2 && isequal(size(x), [1024, 8]) && ~all(x(:) == 0) && ischar(l) && ~isempty(l) && ...
    (isnumeric(x) || isa(x, 'dlarray')), {trainFeatures.embedding}, {trainFeatures.label}));
validVal = find(cellfun(@(x, l) ~isempty(x) && ~any(isnan(x(:))) && ~any(isinf(x(:))) && ...
    ndims(x) == 2 && isequal(size(x), [1024, 8]) && ~all(x(:) == 0) && ischar(l) && ~isempty(l) && ...
    (isnumeric(x) || isa(x, 'dlarray')), {valFeatures.embedding}, {valFeatures.label}));
validTest = find(cellfun(@(x, l) ~isempty(x) && ~any(isnan(x(:))) && ~any(isinf(x(:))) && ...
    ndims(x) == 2 && isequal(size(x), [1024, 8]) && ~all(x(:) == 0) && ischar(l) && ~isempty(l) && ...
    (isnumeric(x) || isa(x, 'dlarray')), {testFeatures.embedding}, {testFeatures.label}));
fprintf('Train samples: %d, Valid embeddings: %d\n', numTrain, length(validTrain));
fprintf('Val samples: %d, Valid embeddings: %d\n', numVal, length(validVal));
fprintf('Test samples: %d, Valid embeddings: %d\n', numTest, length(validTest));

% Log invalid samples
invalidTrain = setdiff(1:numTrain, validTrain);
invalidVal = setdiff(1:numVal, validVal);
invalidTest = setdiff(1:numTest, validTest);
if ~isempty(invalidTrain) || ~isempty(invalidVal) || ~isempty(invalidTest)
    warning('⚠️ Invalid embeddings or labels detected: Train=%d, Val=%d, Test=%d', ...
        length(invalidTrain), length(invalidVal), length(invalidTest));
    for i = 1:min(5, length(invalidTrain))
        fprintf('Invalid Train Sample %d: Label=%s, EmbeddingSize=[%s]\n', ...
            invalidTrain(i), char(trainFeatures(invalidTrain(i)).label), ...
            num2str(size(trainFeatures(invalidTrain(i)).embedding)));
    end
    for i = 1:min(5, length(invalidVal))
        fprintf('Invalid Val Sample %d: Label=%s, EmbeddingSize=[%s]\n', ...
            invalidVal(i), char(valFeatures(invalidVal(i)).label), ...
            num2str(size(valFeatures(invalidVal(i)).embedding)));
    end
    for i = 1:min(5, length(invalidTest))
        fprintf('Invalid Test Sample %d: Label=%s, EmbeddingSize=[%s]\n', ...
            invalidTest(i), char(testFeatures(invalidTest(i)).label), ...
            num2str(size(testFeatures(invalidTest(i)).embedding)));
    end
end

% Check if enough valid samples
if length(validTrain) < 10 || length(validVal) < 2 || length(validTest) < 2
    error('❌ Too few valid samples: Train=%d, Val=%d, Test=%d. Consider re-running extractYAMNetFeatures.m.', ...
        length(validTrain), length(validVal), length(validTest));
end

%% === Determine Sequence Length ===
fprintf("📏 Computing sequence lengths...\n");
trainLengths = cellfun(@(x) size(x, 2), {trainFeatures(validTrain).embedding});
valLengths = cellfun(@(x) size(x, 2), {valFeatures(validVal).embedding});
testLengths = cellfun(@(x) size(x, 2), {testFeatures(validTest).embedding});
maxLength = max([trainLengths, valLengths, testLengths]);
fprintf('Max sequence length: %d frames\n', maxLength);
targetT = min(maxLength, 20); % Cap at 20
fprintf('Selected target sequence length: %d frames\n', targetT);

%% === Prepare Sequences ===
fprintf("🧪 Preparing sequences...\n");

% Helper function to prepare sequences
toSeq = @(features, targetT) deal( ...
    cellfun(@(x) padToLength(convertToSingle(x), targetT), {features.embedding}, 'UniformOutput', false), ...
    categorical({features.label}));

[XTrain, YTrain] = toSeq(trainFeatures(validTrain), targetT);
[XVal, YVal] = toSeq(valFeatures(validVal), targetT);
[XTest, YTest] = toSeq(testFeatures(validTest), targetT);

% Ensure N-by-1 cell arrays
XTrain = reshape(XTrain, [], 1);
XVal = reshape(XVal, [], 1);
XTest = reshape(XTest, [], 1);

% Validate sequence dimensions and contents
fprintf("🔍 Validating sequence dimensions...\n");
validSeqTrain = true(numel(XTrain), 1);
for i = 1:numel(XTrain)
    if ~iscell(XTrain) || ~isvector(XTrain) || size(XTrain, 2) ~= 1
        error('XTrain is not an N-by-1 cell array: Size=[%s]', num2str(size(XTrain)));
    end
    isEmpty = isempty(XTrain{i});
    isNumeric = isnumeric(XTrain{i});
    isCorrectSize = isequal(size(XTrain{i}), [1024, targetT]);
    hasNaN = any(isnan(XTrain{i}(:)));
    hasInf = any(isinf(XTrain{i}(:)));
    isAllZero = all(XTrain{i}(:) == 0);
    isSingle = strcmp(class(XTrain{i}), 'single');
    if isEmpty || ~isNumeric || ~isCorrectSize || hasNaN || hasInf || isAllZero || ~isSingle
        fprintf('Invalid Train Sequence %d: Empty=%d, Numeric=%d, Size=[%s], NaN=%d, Inf=%d, AllZero=%d, Type=%s\n', ...
            i, isEmpty, isNumeric, num2str(size(XTrain{i})), hasNaN, hasInf, isAllZero, class(XTrain{i}));
        validSeqTrain(i) = false;
    end
end
validSeqVal = true(numel(XVal), 1);
for i = 1:numel(XVal)
    if ~iscell(XVal) || ~isvector(XVal) || size(XVal, 2) ~= 1
        error('XVal is not an N-by-1 cell array: Size=[%s]', num2str(size(XVal)));
    end
    isEmpty = isempty(XVal{i});
    isNumeric = isnumeric(XVal{i});
    isCorrectSize = isequal(size(XVal{i}), [1024, targetT]);
    hasNaN = any(isnan(XVal{i}(:)));
    hasInf = any(isinf(XVal{i}(:)));
    isAllZero = all(XVal{i}(:) == 0);
    isSingle = strcmp(class(XVal{i}), 'single');
    if isEmpty || ~isNumeric || ~isCorrectSize || hasNaN || hasInf || isAllZero || ~isSingle
        fprintf('Invalid Val Sequence %d: Empty=%d, Numeric=%d, Size=[%s], NaN=%d, Inf=%d, AllZero=%d, Type=%s\n', ...
            i, isEmpty, isNumeric, num2str(size(XVal{i})), hasNaN, hasInf, isAllZero, class(XVal{i}));
        validSeqVal(i) = false;
    end
end
validSeqTest = true(numel(XTest), 1);
for i = 1:numel(XTest)
    if ~iscell(XTest) || ~isvector(XTest) || size(XTest, 2) ~= 1
        error('XTest is not an N-by-1 cell array: Size=[%s]', num2str(size(XTest)));
    end
    isEmpty = isempty(XTest{i});
    isNumeric = isnumeric(XTest{i});
    isCorrectSize = isequal(size(XTest{i}), [1024, targetT]);
    hasNaN = any(isnan(XTest{i}(:)));
    hasInf = any(isinf(XTest{i}(:)));
    isAllZero = all(XTest{i}(:) == 0);
    isSingle = strcmp(class(XTest{i}), 'single');
    if isEmpty || ~isNumeric || ~isCorrectSize || hasNaN || hasInf || isAllZero || ~isSingle
        fprintf('Invalid Test Sequence %d: Empty=%d, Numeric=%d, Size=[%s], NaN=%d, Inf=%d, AllZero=%d, Type=%s\n', ...
            i, isEmpty, isNumeric, num2str(size(XTest{i})), hasNaN, hasInf, isAllZero, class(XTest{i}));
        validSeqTest(i) = false;
    end
end

% Filter valid sequences
XTrain = XTrain(validSeqTrain);
YTrain = YTrain(validSeqTrain);
XVal = XVal(validSeqVal);
YVal = YVal(validSeqVal);
XTest = XTest(validSeqTest);
YTest = YTest(validSeqTest);
fprintf('✅ Filtered: %d train, %d val, %d test valid sequences\n', ...
    numel(XTrain), numel(XVal), numel(XTest));
fprintf('🧩 All sequences padded to [%d x %d]\n', 1024, targetT);
fprintf('👀 Sample shape check: [%d x %d]\n', size(XTrain{1}));
fprintf('XTrain size: [%d x %d], YTrain length: %d\n', size(XTrain), length(YTrain));

% Check if enough valid sequences
if numel(XTrain) < 10 || numel(XVal) < 2 || numel(XTest) < 2
    error('❌ Too few valid sequences: Train=%d, Val=%d, Test=%d. Consider re-running extractYAMNetFeatures.m.', ...
        numel(XTrain), numel(XVal), numel(XTest));
end

% Validate cell array structure
if ~iscell(XTrain) || ~isvector(XTrain) || size(XTrain, 2) ~= 1
    error('XTrain is not an N-by-1 cell array: Size=[%s]', num2str(size(XTrain)));
end
if length(XTrain) ~= length(YTrain)
    error('XTrain and YTrain length mismatch: XTrain=%d, YTrain=%d', length(XTrain), length(YTrain));
end

%% === Define LSTM Network ===
inputSize = 1024; % YAMNet embedding size
numHiddenUnits = 128;
numClasses = numel(categories(YTrain));

layers = [
    sequenceInputLayer(inputSize)
    lstmLayer(numHiddenUnits, 'OutputMode', 'last')
    dropoutLayer(0.2)
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer
];

options = trainingOptions('adam', ...
    'MaxEpochs', 15, ...
    'MiniBatchSize', 32, ...
    'SequenceLength', targetT, ...
    'ValidationData', {XVal, YVal}, ...
    'ValidationFrequency', 30, ...
    'ValidationPatience', Inf, ... % Disable early stopping
    'Shuffle', 'every-epoch', ...
    'Verbose', 1, ...
    'Plots', 'training-progress');

%% === Train LSTM ===
fprintf("🧠 Training LSTM model...\n");
try
    lstmModel = trainNetwork(XTrain, YTrain, layers, options);
catch e
    error('❌ Training failed: %s', e.message);
end

%% === Save Model ===
if ~exist('models', 'dir')
    mkdir('models');
end
save('models/lstm_model_yamnet.mat', 'lstmModel');
fprintf("✅ LSTM model saved to models/lstm_model_yamnet.mat\n");

%% === Evaluate ===
fprintf("📈 Evaluating on test set...\n");
YPred = classify(lstmModel, XTest, 'MiniBatchSize', 32);

% Compute and print accuracy once
acc = sum(YPred == YTest) / numel(YTest);
fprintf("🎯 Final Test Accuracy: %.2f%%\n", acc * 100);

% Generate confusion matrix in a separate step to avoid interference
figure('Visible', 'off'); % Create figure off-screen to prevent UI refresh issues
cm = confusionchart(YTest, YPred);
title('LSTM Confusion Matrix (YAMNet Embeddings)');
fprintf("📊 Confusion Matrix generated. Check figure window for details.\n");

%% === Helper: Pad or Truncate to targetT ===
function x = padToLength(x, targetT)
    [D, T] = size(x); % Expect x to be [1024 x T]
    if T == targetT
        return;
    elseif T > targetT
        x = x(:, 1:targetT);  % Truncate
    else
        x = [x, zeros(D, targetT - T)];  % Pad with zeros
    end
end

%% === Helper: Convert to Single ===
function x = convertToSingle(x)
    if isa(x, 'dlarray')
        x = extractdata(x); % Convert dlarray to numeric
    end
    if ~isnumeric(x)
        error('Non-numeric embedding detected: Type=%s', class(x));
    end
    x = single(x); % Cast to single precision
end