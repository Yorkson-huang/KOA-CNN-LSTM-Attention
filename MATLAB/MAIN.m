%% Load feature data and corresponding wind-speed targets
rawData = xlsread('Data.xlsx');
Features = rawData(1:18,:);   %% 18 features across 75 days x 24 hours (1800 time steps)
WindData = rawData(19,:);     %% Wind-speed target for the same 1800 time steps

%% Reshape into 4-D tensors to mimic MATLAB Deep Learning Toolbox expectations
LP_Features = double(reshape(Features,18,24,1,75)); %% 18x24x1x75: features per day
LP_WindData = double(reshape(WindData,24,1,1,75)); %% 24x1x1x75: targets per day

%% Convert to cell arrays indexed by day
numDays = 75;
for i = 1:numDays
    FeaturesData{1,i} = LP_Features(:,:,1,i);
end
for i = 1:numDays
    RealData{1,i} = LP_WindData(:,:,1,i);
end

%% Train/test split
XTrain = FeaturesData(:,1:73); %% Training inputs: days 1-73
YTrain = RealData(:,2:74);     %% Training targets: days 2-74 (one-day look-ahead)
XTest  = cell2mat(FeaturesData(:,74)); %% Test input: day 74
YTest  = cell2mat(RealData(:,75));     %% Test target: day 75

%% Fixed hyper-parameters
learningRate = 0.01;
kernelSize   = 3;
numNeurons   = 32;

%% Build the network
lgraph = [
    sequenceInputLayer([18 24 1],"Name","sequence")
    convolution2dLayer(kernelSize,3,"Name","conv","Padding","same")
    batchNormalizationLayer("Name","batchnorm")
    reluLayer("Name","relu")
    maxPooling2dLayer([3 3],"Name","maxpool","Padding","same")
    flattenLayer("Name","flatten")
    lstmLayer(numNeurons,"Name","lstm")
    fullyConnectedLayer(24,"Name","fc")
    regressionLayer("Name","regressionoutput")];

%% Training options (SGDM)
options = trainingOptions('sgdm', ...
    'MaxEpochs',400, ...
    'GradientThreshold',1, ...
    'ExecutionEnvironment','cpu', ...
    'InitialLearnRate',learningRate, ...
    'LearnRateSchedule','none', ...
    'Shuffle','every-epoch', ...
    'SequenceLength',24, ...
    'MiniBatchSize',15, ...
    'Verbose',true);

%% Train
[net,info] = trainNetwork(XTrain,YTrain,lgraph,options);

%% Evaluate on the held-out day
YPredicted = net.predict(XTest);
errorTerm = YPredicted - YTest;
[~,numSamples] = size(YTest);
SSE  = sum(errorTerm.^2);
MAE  = sum(abs(errorTerm))/numSamples;
MSE  = errorTerm*errorTerm'/numSamples;
RMSE = sqrt(MSE);
MAPE = mean(abs(errorTerm./mean(YTest)));
R    = corrcoef(YTest,YPredicted);
Rval = R(1,2);

fprintf('Current batch MAPE: %f\n', MAPE);
