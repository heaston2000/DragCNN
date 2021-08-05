function [X,Y] = minibatchProcess(XCell,YCell)
    
    % Extract image data from cell and concatenate, possible error?
    X = cat(4,XCell{:});
    size(X);
    % Extract angle data from cell and concatenate
    Y = cat(4,YCell{:});
    size(Y);
    %X = reshape(X, [128 128 1 64]);
    
    
    
end