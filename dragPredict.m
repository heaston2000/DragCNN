function [predictions, diff] = dragPredict(net, dsTest, miniBatchSize)
% Function to predict the outputs of the 


mbqTest = minibatchqueue(dsTest,...
    "MiniBatchSize",miniBatchSize,...
    "MiniBatchFcn", @minibatchProcess,...
    "MiniBatchFormat","SSCB");

numObservations = numel(dsTest);
predictions = [];
diff = [];

while hasdata(mbqTest)
    [dlXTest,dlYTest] = next(mbqTest);
    dlYpred = predict(net, dlXTest);
    predBatch = extractdata(dlYpred);
    
    if size(predBatch, 2) == miniBatchSize
        
        predictions = cat(4, predictions, predBatch);

        diffBatch = predBatch - dlYTest;
    
        diff = cat(4, diff, extractdata(gather(diffBatch)));
        
    end
end

    
end

