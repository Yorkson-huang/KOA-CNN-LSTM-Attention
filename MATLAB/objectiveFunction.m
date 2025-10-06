function [R,tsmvalue,net,info] = objectiveFunction(x)
%% Load feature matrix and wind-speed targets
rawData = xlsread('Data.xlsx');
Features   = rawData(1:18,:);  %% 18 feature series (75 days x 24 hours)
Wind_data  = rawData(19,:);    %% Wind-speed targets for the same horizon

%% Reshape into 4-D tensors for MATLAB deep learning layers
LP_Features = double(reshape(Features,18,24,1,75));  %% 18x24x1x75
LP_WindData = double(reshape(Wind_data,24,1,1,75));  %% 24x1x1x75

%% Convert to day-indexed cells
numDays = 75;
for i = 1:numDays
    FeaturesData{1,i} = LP_Features(:,:,1,i);
end
for i = 1:numDays
    RealData{1,i} = LP_WindData(:,:,1,i);
end

%% Split into training and testing sets
XTrain = FeaturesData(:,1:73); %% days 1-73
YTrain = RealData(:,2:74);     %% days 2-74 (one-day shift)
XTest  = cell2mat(FeaturesData(:,74));
YTest  = cell2mat(RealData(:,75));

%% Map optimization variables to hyper-parameters
learning_rate = x(1);
kernelSize    = round(x(2));
numNeurons    = round(x(3));

%% Build the KOA-tuned network
lgraph = [
    sequenceInputLayer([18 24 1],"Name","sequence")
    convolution2dLayer(kernelSize,3,"Name","conv","Padding","same")
    batchNormalizationLayer("Name","batchnorm")
    reluLayer("Name","relu")
    maxPooling2dLayer([3 3],"Name","maxpool","Padding","same")
    flattenLayer("Name","flatten")
    lstmLayer(numNeurons,"Name","lstm")
    selfAttentionLayer(1,24,"Name","selfattention")
    fullyConnectedLayer(24,"Name","fc")
    regressionLayer("Name","regressionoutput")
    ];

%% Training configuration
options = trainingOptions('sgdm', ...
    'MaxEpochs',400, ...
    'GradientThreshold',1, ...
    'ExecutionEnvironment','cpu', ...
    'InitialLearnRate',learning_rate, ...
    'LearnRateSchedule','none', ...
    'Shuffle','every-epoch', ...
    'SequenceLength',24, ...
    'MiniBatchSize',15, ...
    'Verbose',true);

%% Train the model
[net,info] = trainNetwork(XTrain,YTrain,lgraph,options);

%% Evaluate on the hold-out day
YPredicted = net.predict(XTest);
tsmvalue    = YPredicted;

%% Error metrics
errorTerm = YPredicted - YTest;
[~,len] = size(YTest);
SSE1  = sum(errorTerm.^2);
MAE1  = sum(abs(errorTerm))/len;
MSE1  = errorTerm*errorTerm'/len;
RMSE1 = sqrt(MSE1);
MAPE1 = mean(abs(errorTerm./mean(YTest)));
r     = corrcoef(YTest,YPredicted);
R1    = r(1,2);
R     = MAPE1;

fprintf('Current batch MAPE: %f\n', R);
end
