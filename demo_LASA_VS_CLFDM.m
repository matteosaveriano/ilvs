%% 
% Run CLF-DM approach on the LASA Visual Servoing HandWriting Dataset
%
% The original code for CLF-DM is available at: 
% https://bitbucket.org/khansari/clfdm/src/master/

%%
clear;
close all;

addpath(genpath('lib'))
addpath(genpath('datasets'))

%% Setting parameters of the Control Lyapunov Function
Vxf0.d = 6; % Space dimention
Vxf0.w = 1e-4; %A positive scalar weight regulating the priority between the 
               %two objectives of the opitmization. Please refer to the
               %page 7 of the paper for further information.

% A set of options that will be passed to the solver. Please type 
% 'doc preprocessDemos' in the MATLAB command window to get detailed
% information about other possible options.
options.tol_mat_bias = 10^-1; % a very small positive scalar to avoid
                              % having a zero eigen value in matrices P^l [default: 10^-15]
                              
options.display = [];          % An option to control whether the algorithm
                              % displays the output of each iterations [default: true]
                              
options.tol_stopping=10^-10;  % A small positive scalar defining the stoppping
                              % tolerance for the optimization solver [default: 10^-10]

options.max_iter = 500;       % Maximum number of iteration for the solver [default: i_max=1000]
options.i_max = 500;

options.optimizePriors = true;% This is an added feature that is not reported in the paper. In fact
                              % the new CLFDM model now allows to add a prior weight to each quadratic
                              % energy term. IF optimizePriors sets to false, unifrom weight is considered;
                              % otherwise, it will be optimized by the sovler.
                              
options.upperBoundEigenValue = true; %This is also another added feature that is impelemnted recently.
                                     %When set to true, it forces the sum of eigenvalues of each P^l 
                                     %matrix to be equal one. 

%% Create a dummy pinhole camera
% Camera matrix
KP = zeros(3,4);  
KP(1,1) = 800;
KP(2,2) = 800;
KP(3,3) = 1;
KP(1:2,3) = 512;
          
%% Learning and reproduction loop
motionClass = 7; % Which of the 30 motions to use
demoNum = 1:3; % Consider only the first 3 demonstrations
samplingRate = 10; % Resample the trajectory to make learning faster

%% Create training data
% Load demonstrations
[demos, name] = load_LASA_VS_models('LASA_HandWriting_VS/', motionClass);
demoSize = size(demos{1}.pos, 2);
sInd = [1:samplingRate:demoSize demoSize]; % Use only points at 'sInd'
        
% Retrieve point grid, goal depth, and sampling time
P = demos{1}.point_grid;
depthGoal = demos{1}.depth_goal; % Final depth

% Initialize some variables to store results
pos = [];
vel = [];
feat = [];
featStarAll = [];
poseStarAll = [];
poseInitAll = [];
for i=demoNum
    dt_ = samplingRate*demos{i}.dt;
    cameraPose{i} = demos{i}.pos(:,sInd(1:end));
    cameraVel{i} = [diff(cameraPose{i}, [], 2)./dt_, zeros(6,1)];
    cameraVel{i}(1:3,:) = cameraVel{i}(1:3,:);
    cameraFeat{i} = demos{i}.feat(:,sInd(1:end));

    pos = [pos; cameraPose{i}'];
    vel = [vel; cameraVel{i}'];
    feat = [feat; cameraFeat{i}'];

    featStarAll = [featStarAll; demos{i}.feat(:,end)'];
    poseStarAll = [poseStarAll; demos{i}.pos(:,end)'];
    poseInitAll = [poseInitAll; demos{i}.pos(:,1)'];
end
    
allDemoLen = size(feat, 1);
featStar = mean(featStarAll);
poseStar = mean(poseStarAll);
featErr = feat - repmat(featStar, allDemoLen, 1);
poseErr = zeros(allDemoLen, 6);
uErr = zeros(allDemoLen, 6);

% Compute image Jacobian at goal
Lgoal = visualJacobianMatrix(reshape(featStar, 2, 4), depthGoal, KP);
LpGoal = pinv(Lgoal);

for i=1:allDemoLen
    % Compute Lp*e for each point
    poseErr(i,:) = (LpGoal*featErr(i,:).').';
    % Desired control input is vel
    uErr(i,:) = vel(i,:);
end
    
%% Fit GMM model
Data = [poseErr'; uErr'];
in = 1:6;
out = 7:12; 
nbStates = 11; % Gaussian components

[Priors, Mu, Sigma] = EM_init_kmeans(Data, nbStates);
[Priors, Mu, Sigma] = EM(Data, Priors, Mu, Sigma);   
    
gmrHandle = @(x) GMR(Priors, Mu, Sigma, x, in, out);
    
%% Train CLF
Vxf0.L = 1; % Number of asymmetric components, L>=0
Vxf0.Priors = ones(Vxf0.L+1,1);
Vxf0.Priors = Vxf0.Priors/sum(Vxf0.Priors);
Vxf0.Mu = zeros(Vxf0.d,Vxf0.L+1);
for l=1:Vxf0.L+1
    Vxf0.P(:,:,l) = eye(Vxf0.d);
end

% Solving the optimization
 Vxf = learnEnergy(Vxf0, Data, options);
% Function handles for simulation
rho0 = 0.01;
kappa0 = 0.001;
    
uClfHandle= @(y) DS_stabilizer(y, gmrHandle, Vxf, rho0, kappa0);

%% Simulate
demoLen = allDemoLen/length(demoNum); % Length of the single demonstration
sStar = featStar.'; % Goal in feature space
Tcam = eye(4); % Initial pose of the dummy camera
figure;
for d=demoNum
    pClfdm = zeros(3, demoLen);
    pClfdm(:,1) = poseInitAll(d,1:3)';
    vClfdm = zeros(6, demoLen);
    % Update camera position (orientation is fixed)   
    Tcam(1:3,4) = [pClfdm(1,1); pClfdm(2,1); pClfdm(3,1)];
    
    error = zeros(demoLen-1, 8);
    sClfdm = zeros(8, demoLen-1);    
    for i=1:demoLen-1 % Run a bit longer to test for convergence
        % Project to image plane
        sCurr = cameraPoseToImagePoints(Tcam, P, KP);
        % Store features for plotting
        sClfdm(:,i) = reshape(sCurr, 8, 1);
        % Compute image error
        error(i,:) = reshape(sCurr, 8, 1) - sStar;
        % Compute CLFDM velocity
        vClfdm(:,i) = uClfHandle(LpGoal*error(i,:)');

        % Update camera position (orientation is fixed)
        Tcam(1:3,4) = Tcam(1:3,4) + vClfdm(1:3,i)*dt_;
        pClfdm(:, i+1) = Tcam(1:3,4);
    end

    %% Plot results
    colorGreen = [0 127 0]/255;
    
    h1 = subplot(1,2,1);
    title('Cartesian position')
    plot(pClfdm(1,:), pClfdm(2,:), 'color', colorGreen, 'LineWidth', 3)
    hold on
    
    plot(cameraPose{d}(1,1), cameraPose{d}(2,1), 'ko', 'MarkerSize', 15, 'MarkerFaceColor', 'k')
    plot(cameraPose{d}(1,:), cameraPose{d}(2,:), 'k--', 'LineWidth', 3)
    plot(cameraPose{d}(1,end), cameraPose{d}(2,end), 'kx', 'MarkerSize', 15, 'LineWidth', 3)
    
    set(h1, 'YDir','reverse');
    ax = h1;
    ax.XLim = ax.XLim + [-.1, .1];
    ax.YLim = ax.YLim + [-.1, .1];

    h2 = subplot(1,2,2);
    box on
    hold on
    title('Features position')
 
    plot(sClfdm(1,:), sClfdm(2,:), 'color', colorGreen, 'LineWidth', 3)
    plot(sClfdm(3,:), sClfdm(4,:), 'color', colorGreen, 'LineWidth', 3)
    plot(sClfdm(5,:), sClfdm(6,:), 'color', colorGreen, 'LineWidth', 3)
    plot(sClfdm(7,:), sClfdm(8,:), 'color', colorGreen, 'LineWidth', 3)

    sDem = cameraFeat{d};

    plot(sDem(1,1), sDem(2,1), 'ko', 'MarkerSize', 15, 'MarkerFaceColor', 'k')
    plot(sDem(3,1), sDem(4,1), 'ko', 'MarkerSize', 15, 'MarkerFaceColor', 'k')
    plot(sDem(5,1), sDem(6,1), 'ko', 'MarkerSize', 15, 'MarkerFaceColor', 'k')
    plot(sDem(7,1), sDem(8,1), 'ko', 'MarkerSize', 15, 'MarkerFaceColor', 'k')
    
    plot(sDem(1,:), sDem(2,:), 'k--', 'LineWidth', 3)
    plot(sDem(3,:), sDem(4,:), 'k--', 'LineWidth', 3)
    plot(sDem(5,:), sDem(6,:), 'k--', 'LineWidth', 3)
    plot(sDem(7,:), sDem(8,:), 'k--', 'LineWidth', 3)

    plot(sStar(1,:), sStar(2,:), 'kx', 'MarkerSize', 15, 'LineWidth', 3)
    plot(sStar(3,:), sStar(4,:), 'kx', 'MarkerSize', 15, 'LineWidth', 3)
    plot(sStar(5,:), sStar(6,:), 'kx', 'MarkerSize', 15, 'LineWidth', 3)
    plot(sStar(7,:), sStar(8,:), 'kx', 'MarkerSize', 15, 'LineWidth', 3)
    
    set(h2, 'YDir','reverse');
    ax = h2;
    ax.XLim = ax.XLim + [-10, 10];
    ax.YLim = ax.YLim + [-10, 10];
end