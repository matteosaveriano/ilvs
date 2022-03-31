clear;
close all;

addpath(genpath('../ds_handles/'));
addpath(genpath('../LASA_hand_writing_dataset/'));
addpath(genpath('../utilities/'));

modelPath = '../LASA_hand_writing_dataset/';

%% Putting CLFDM and GMR library in the MATLAB Path
if isempty(regexp(path,['CLFDM_lib' pathsep], 'once'))
    addpath([pwd, '/CLFDM_lib']);    % add SEDS dir to path
end

%% Setting parameters of the Control Lyapunov Function
Vxf0.d = 2;
Vxf0.w = 1e-4; %A positive scalar weight regulating the priority between the 
               %two objectives of the opitmization. Please refer to the
               %page 7 of the paper for further information.

% A set of options that will be passed to the solver. Please type 
% 'doc preprocess_demos' in the MATLAB command window to get detailed
% information about other possible options.
options.tol_mat_bias = 10^-1; % a very small positive scalar to avoid
                              % having a zero eigen value in matrices P^l [default: 10^-15]
                              
options.display = [];          % An option to control whether the algorithm
                              % displays the output of each iterations [default: true]
                              
options.tol_stopping=10^-10;  % A small positive scalar defining the stoppping
                              % tolerance for the optimization solver [default: 10^-10]

options.max_iter = 1500;       % Maximum number of iteration for the solver [default: i_max=1000]
options.i_max = 1500;

options.optimizePriors = true;% This is an added feature that is not reported in the paper. In fact
                              % the new CLFDM model now allows to add a prior weight to each quadratic
                              % energy term. IF optimizePriors sets to false, unifrom weight is considered;
                              % otherwise, it will be optimized by the sovler.
                              
options.upperBoundEigenValue = true; %This is also another added feature that is impelemnted recently.
                                     %When set to true, it forces the sum of eigenvalues of each P^l 
                                     %matrix to be equal one. 
                                     
%% Demonstrations downsampling parameters
demoNum = [1 2 3];
samplingRate = 10;

%% Simulation parameters
opt_sim.i_max = 500;
opt_sim.tol   = 0.1;
opt_sim.plot  = false;

%% Learning loop
clfdmTrainingTime = zeros(1,26);
gmmTrainingTime   = zeros(1,26);
for modIt=1:26
    %% Load demonstrations
    [demos, dt]   = load_LASA_models(modelPath, modIt);
    simOptions.dt = samplingRate*dt;
    opt_sim.dt    = simOptions.dt;
    demoSize = size(demos{1}.pos, 2);
    sInd = [1:samplingRate:demoSize, demoSize];

    %% Store motion demonstrations
    xDemo  = [];
    xdDemo = [];

    % Create demonstrations
    for demoIt=demoNum   
        xCurr = demos{demoIt}.pos(:,sInd);
        vCurr = [diff(xCurr,[],2)./simOptions.dt [0;0]];
        
        xDemo  = [xDemo xCurr];
        xdDemo = [xdDemo vCurr];
        
        x0(:,demoIt) = demos{demoIt}.pos(:,1);
    end
    trainingData = [xDemo; xdDemo];


    timeGMM = [];
    for nbStates = 4:7 % Find best number of components
        %% Train GMM/GMR
        tic;
        [Priors, Mu, Sigma] = EM_init_kmeans(trainingData, nbStates);
        [Priors, Mu, Sigma] = EM(trainingData, Priors, Mu, Sigma);
        timeGMM = [timeGMM toc];
        
        %% Train CLF       
        % Estimating an initial guess for the Lyapunov function
       % timeCLF = [];
        for asymComp = 0:4
            tic;
            Vxf0.L = asymComp; % Number of asymmetric components, L>=0
            Vxf0.Priors = ones(Vxf0.L+1,1);
            Vxf0.Priors = Vxf0.Priors/sum(Vxf0.Priors);
            Vxf0.Mu = zeros(Vxf0.d,Vxf0.L+1);
            for l=1:Vxf0.L+1
                Vxf0.P(:,:,l) = eye(Vxf0.d);
            end

            % Solving the optimization
            Vxf = learnEnergy(Vxf0, trainingData, options);
            timeCLF(nbStates-3, asymComp+1) = toc;
        
            % Function handles for simulation
            rho0 = 1;
            kappa0 = 0.1;
            % 
            in = 1:Vxf.d;
            out = Vxf.d+1:2*Vxf.d;
            handleGMR = @(x) GMR(Priors, Mu, Sigma, x, in, out);
            handleCLF = @(y) DS_stabilizer(y, handleGMR, Vxf, rho0, kappa0);

            %% Swept Area and Vrmse distances
            for demoIt=demoNum
                pos = Simulation(x0(:,demoIt), [], handleCLF, opt_sim);

                tmpSEA(nbStates-3, asymComp+1, demoIt) = sweptAreaError(pos, demos{demoIt}.pos(:,sInd(1:end)));
                demoX = demos{demoIt}.pos(:,sInd(1:end));
                demoV = [diff(demoX,[],2)./simOptions.dt [0;0]];
                tmpVRMSE(nbStates-3, asymComp+1, demoIt) = velocityRMSE_CLFDM(handleCLF, demoX, demoV);
            end
        end
    end
    
    % Find and store only best results
    [S, C, D] = size(tmpSEA);
    meanSEA = zeros(S, C);
    for s=1:S
        for c=1:C
            meanSEA(s,c) = mean(tmpSEA(s,c,:));
        end
    end
    
    minMeanSEA = min(meanSEA(:));
    [indS,indC] = find(meanSEA==minMeanSEA);
    
    allSEA(modIt,:)   = tmpSEA(indS(1), indC(1), :); % Take the first index in case of multiple minima
    allVRMSE(modIt,:) = tmpVRMSE(indS(1), indC(1), :);
    clfdmTrainingTime(modIt) = timeCLF(indS(1),indC(1));
    gmmTrainingTime(modIt)   = timeGMM(indS(1));
end

%% Get global results
disp(['Mean distance: ' num2str(mean(mean(allSEA,2)))]);
disp(['Max distance: ' num2str(max(mean(allSEA,2)))]);