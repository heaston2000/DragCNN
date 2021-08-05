function [dldragNet] = TrainDragModel(dldragNet, miniBatchSize, numEpochs, dsTrain)
% Given Deep Learning Architecture, train a network

mbq = minibatchqueue(dsTrain,...
    'MiniBatchSize',miniBatchSize,...
    'MiniBatchFormat',{'SSCB', 'SSCB'},...
    'MiniBatchFcn', @minibatchProcess);

% Adam Optimizer
trailingAvg = [];
trailingAvgSq = [];
iteration = 0;
start = tic;
learnrate = 0.0001;
gradDecay = 0.005;

% Set the plots

plots = "training-progress";

% TRAIN THE MODEL:
if plots == "training-progress"
    figure
    lineLossTrain = animatedline('Color',[0.85 0.325 0.098]);
    ylim([0 inf])
    xlabel("Iteration")
    ylabel("Loss")
    grid on
end

% Loop over epochs.
for epoch = 1:numEpochs
    
    % Shuffle data.
    shuffle(mbq)
    
    % Loop over mini-batches
    while hasdata(mbq) 
        
        iteration = iteration + 1;
        
        [dlX,dlY] = next(mbq);
        
        target = dlY;
                       
        % Evaluate the model gradients, state, and loss using dlfeval and the
        % modelGradients function.
        [gradients,state,loss] = dlfeval(@modelGradients, dldragNet, dlX, target);
        
        dldragNet.State = state;
        
        % Update the network parameters using the Adam optimizer.
        [dldragNet,trailingAvg,trailingAvgSq] = adamupdate(dldragNet,gradients, ...
            trailingAvg,trailingAvgSq,iteration, learnrate, gradDecay);
        
        % Display the training progress.
        if plots == "training-progress"
            D = duration(0,0,toc(start),'Format','hh:mm:ss');
            addpoints(lineLossTrain,iteration,double(gather(extractdata(loss))))
            title("Epoch: " + epoch + ", Elapsed: " + string(D))
            drawnow
        end
    end
end

end



