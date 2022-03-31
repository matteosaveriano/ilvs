function pos = reshapeCLF(DSHandle, LRDSHandle, x0, simOptions)
    for demoIt=1:size(x0,2)
        it = 1;
        xCurrMod = x0(:,demoIt);
        itM   = 0;
        while(it<simOptions.iterNum && norm(xCurrMod-simOptions.goal)>simOptions.tol)
            % Store data for plotting
            xM(:,it) = xCurrMod;
            
            w = 1;
            if(it>simOptions.Tmax || norm(xCurrMod-simOptions.goal)<simOptions.tol)
                w = exp(-13*itM*simOptions.dt);
                itM = itM + 1;
            end

%             alpha     = computeGpWeights(GPR(1), xCurrMod', truncValues);
%             par.xd(1,1) = alpha' * GPR(1).data.out;
%             par.xd(2,1) = alpha' * GPR(2).data.out;

            % Compute next modulated position
            if(isempty(LRDSHandle))
                vCurrMod = DSHandle(xCurrMod);
            else
                vCurrMod = DSHandle(xCurrMod) + w*LRDSHandle(xCurrMod);
            end
            xCurrMod = xCurrMod + vCurrMod * simOptions.dt;

            it = it + 1;
            
            if(it > 39)
                eee=1;
            end
        end
        
        if(simOptions.plotResult)
            figure(1);
            hold on;
            plot(xM(1,1:1:end), xM(2,1:1:end), 'k','Linewidth', 3)
            plot(xM(1,end), xM(2,end), 'k.','Linewidth', 2,'MarkerSize', 50)
        end
        
        pos{demoIt} = xM;
        
        clear xM
    end
end