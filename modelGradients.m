function [gradients, state, loss] = modelGradients(dldragNet, dlX, target) 
            % the predictions Y and the training targets T.
            
            [dlY, state] = forward(dldragNet,dlX);
            dummy = zeros(1, 1, 1, size(dlX, 4));
            dummy(1,1,:,:) = dlY;
            dlY = dummy;
            dlY = dlarray(dlY, 'SSCB');
            
            % Define dimensions
            %A = size(out_p);
            %Channel_weights = ones(1,1,3,1);
            %Channel_weights(:,:,1,:) = 0.1153;
            %Channel_weights(:,:,2,:) = 0.0175;
            %Channel_weights(:,:,3,:) = 0.0135;
            
            
            % Calculate errors.
            %loss_u = (dlY(:,:,1,:)-target(:,:,1,:)).^2;
            %loss_v = (dlY(:,:,2,:)-target(:,:,2,:)).^2;
            %loss_p = abs(dlY(:,:,3,:)-target(:,:,3,:));
            %totLoss = (loss_u + loss_v + loss_p)./Channel_weights;
            loss = dlarray(mse(dlY, target));
            
            gradients = dlgradient(loss,dldragNet.Learnables);
end