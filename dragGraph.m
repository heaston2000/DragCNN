function [dragGraph] = dragGraph()
% Creates layer graph specialized for drag prediction
dragGraph = layerGraph();


layers = [
    imageInputLayer([128 128 1],"Name","imageinput", "Normalization", "none")
    convolution2dLayer([3 3],16,"Name","conv_1","Padding","same")
    reluLayer("Name","relu_1")
    convolution2dLayer([3 3],16,"Name","conv_2","Padding","same")
    reluLayer("Name","relu_3")
    maxPooling2dLayer([2 2],"Stride", [2 2],"Name","maxpool_1","Padding","same")
    convolution2dLayer([3 3],16,"Name","conv_3","Padding","same")
    reluLayer("Name","relu_2")
    convolution2dLayer([3 3],16,"Name","conv_4","Padding","same")
    reluLayer("Name","relu_4")
    maxPooling2dLayer([2 2],"Stride", [2 2],"Name","maxpool_2","Padding","same")
    convolution2dLayer([3 3],32,"Name","conv_5","Padding","same")
    reluLayer("Name","relu_5")
    convolution2dLayer([3 3],32,"Name","conv_6","Padding","same")
    reluLayer("Name","relu_6")
    maxPooling2dLayer([2 2],"Stride", [2 2],"Name","maxpool_3","Padding","same")
    convolution2dLayer([3 3],32,"Name","conv_7","Padding","same")
    reluLayer("Name","relu_8")
    convolution2dLayer([3 3],32,"Name","conv_8","Padding","same")
    reluLayer("Name","relu_7")
    maxPooling2dLayer([2 2],"Stride", [2 2],"Name","maxpool_4","Padding","same")
    convolution2dLayer([3 3],32,"Name","conv_9","Padding","same")
    reluLayer("Name","relu_10")
    convolution2dLayer([3 3], 32,"Name","conv_10","Padding","same")
    reluLayer("Name","relu_9")
    maxPooling2dLayer([2 2],"Stride", [2 2], "Name","maxpool_5","Padding","same")
    fullyConnectedLayer(64,"Name","fc")
    reluLayer("Name", "relu_12")
    fullyConnectedLayer(1, "Name", "last")
    regressionLayer("Name", "rOutput")];
 


dragGraph = addLayers(dragGraph, layers);



% Connect layers (Only necessary if structure of graph is not straght line)
%dragGraph = connectLayers(dragGraph, "imageinput", "conv_1");
%dragGraph = connectLayers(dragGraph, "conv_1", "relu_1");
%dragGraph = connectLayers(dragGraph, "relu_1", "conv_2");
%dragGraph = connectLayers(dragGraph, "conv_2", "relu_3");
%dragGraph = connectLayers(dragGraph, "relu_3", "maxpool_1");
%dragGraph = connectLayers(dragGraph, "maxpool_1", "conv_3");
%dragGraph = connectLayers(dragGraph, "conv_3", "relu_2");
%dragGraph = connectLayers(dragGraph, "relu_2", "conv_4");
%dragGraph = connectLayers(dragGraph, "conv_4", "relu_4");
%dragGraph = connectLayers(dragGraph, "relu_4", "maxpool_2");
%dragGraph = connectLayers(dragGraph, "maxpool_2", "conv_5");
%dragGraph = connectLayers(dragGraph, "conv_5", "relu_5");
%dragGraph = connectLayers(dragGraph, "relu_5", "conv_6");
%dragGraph = connectLayers(dragGraph, "conv_6", "relu_6");
%dragGraph = connectLayers(dragGraph, "relu_6", "maxpool_3");
%dragGraph = connectLayers(dragGraph, "maxpool_3", "conv_7");
%dragGraph = connectLayers(dragGraph, "conv_7", "relu_8");
%dragGraph = connectLayers(dragGraph, "relu_8", "conv_8");
%dragGraph = connectLayers(dragGraph, "conv_8", "relu_7");
%dragGraph = connectLayers(dragGraph, "relu_7", "maxpool_4");
%dragGraph = connectLayers(dragGraph, "maxpool_4", "conv_9");
%dragGraph = connectLayers(dragGraph, "conv_9", "relu_10");
%dragGraph = connectLayers(dragGraph, "relu_10", "conv_10");
%dragGraph = connectLayers(dragGraph, "conv_10", "relu_9");
%dragGraph = connectLayers(dragGraph, "relu_9", "maxpool_5");
%dragGraph = connectLayers(dragGraph, "maxpool_5", "fc");
%dragGraph = connectLayers(dragGraph, "fc", "relu_11");

% If using custom training loop:
% dldragNet = dlnetwork(dragGraph);



end

