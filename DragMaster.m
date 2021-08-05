%% Final Drag-Coeffcient Neural Network

XPath = "/Users/heaston/Dropbox/Mac/Documents/MATLAB/ds/shapes";
YPath = "/Users/heaston/Dropbox/Mac/Documents/MATLAB/ds/drags";

%Load and Preprocess the data
[trainX, trainY, testX, testY, valX, valY] = dragPreprocess(XPath, YPath);



%% Create the deep-learning graph architecture as a dlnet object (hopefully):

[dragGraph] = dragGraph();


%% Train the Network

numEpochs = 300;
miniBatchSize = 256;


options = trainingOptions('adam', 'Verbose', true, 'Plots',"training-progress", ...
 "VerboseFrequency",10, "InitialLearnRate", 0.001, "LearnRateSchedule", ...
"piecewise", "Shuffle", "every-epoch", "LearnRateDropFactor", 0.005, 'MaxEpoch', ...
numEpochs, 'MiniBatchSize', 16, 'ValidationData', {valX, valY}, 'ValidationFrequency', 10);
 


net = trainNetwork(trainX, trainY, dragGraph, options)

%% Make Predictions

predictions = predict(net, testX);
diff = abs(predictions - testY);

dragMSE = mean((diff.^2), 'all')

averageLoss = mean(diff, 'all');
averageDrag = mean(testY, 'all');
relativeError = abs(averageLoss / averageDrag)
