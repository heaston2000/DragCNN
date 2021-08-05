function [trainX, trainY, testX, testY, valX, valY] = dragPreprocess(XPath, YPath)
% Load the data into arrays and preprocess the images

dataX = zeros([128 128 1 12000]);
dataY = zeros([1 1 1 12000]);

% maximum = 0; This was used to find the maximum pixel value among all the
% shapes
for i = 0:11999
    %Prepare the input data as grayscale image scaled down by one half
    img = double(im2gray(imread(XPath + "/shape_" + string(i) + ".png"))); % This is where im2gray was
    img = imresize(img, 0.5);
    img = double(img);
    
    
    %tempMax = max(max(img));
    %maximum = max([tempMax maximum]) <-- Pieces of code used to find the
    %maximum pixel value among the images
    img = img./65;  % 65 was the maximum pixel value found among all images
    
    %imshow(img)
    dataX(:,:,:,(i+1)) = img; 
    
        
    %Prepare the output data
    sample = readtable(YPath + "/shape_" + string(i));
    %mistakes = isnan(table2array(sample(:,3)));
    %if any(mistakes)
    %    sample(mistakes, 3) = sample(mistakes, 4);
    %end
    
    drag = double(table2array(sample(40,2)));
    
    dataY(1,1,1,(i+1)) = drag;
    
end


% Partition data into testing and training sets
[TrainInd, ValInd, TestInd] = dividerand(12000, 0.8, 0.1, 0.1);
trainX = dataX(:,:,:,TrainInd);
trainY = dataY(:,:,:,TrainInd);
testX = dataX(:,:,:,TestInd);
testY = dataY(:,:,:,TestInd);
valX = dataX(:,:,:,ValInd);
valY = dataY(:,:,:,ValInd);

%Putting data into datastores (necessary if using custom training loop)
%dsTrainX = arrayDatastore(trainX,'IterationDimension',4);
%dsTrainY = arrayDatastore(trainY,'IterationDimension',4); %Possible source of error
%dsTestX = arrayDatastore(testX, 'IterationDimension', 4);
%dsTestY = arrayDatastore(testY, 'IterationDimension', 4);



end
