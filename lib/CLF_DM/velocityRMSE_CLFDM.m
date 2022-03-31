function [errRMSE, err2] = velocityRMSE_CLFDM(DSHandle, xDemo, vDemo)
    for i =1:length(xDemo)
        if(norm(xDemo(:,i))>1e-3) % Otherwise NaN
            xCurrMod = xDemo(:,i);

            % Compute stabilized velocity
            vCurrMod = DSHandle(xCurrMod);

            % Compute suqared errors
            err2(i) = (vDemo(:,i)-vCurrMod)'*(vDemo(:,i)-vCurrMod);
        end
    end 
    % Return RMSE error
    errRMSE = sqrt(mean(err2)); 
end