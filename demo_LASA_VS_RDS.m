%% 
% Run RDS approach on the LASA Visual Servoing HandWriting Dataset
%
%%
clear;
close all;

addpath(genpath('lib'))
addpath(genpath('datasets'))

%% Create a dummy pinhole camera
% Camera matrix
KP = zeros(3,4);  
KP(1,1) = 800;
KP(2,2) = 800;
KP(3,3) = 1;
KP(1:2,3) = 512;
          
%% Learning and reproduction loop
motionClass = 26; % Which of the 30 motions to use
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

lambda_ = 1; % Classic visual servoing gain

% Compute image Jacobian at goal
Lgoal = visualJacobianMatrix(reshape(featStar, 2, 4), depthGoal, KP);
LpGoal = pinv(Lgoal);

for i=1:allDemoLen
    % Compute Lp*e for each point
    poseErr(i,:) = (LpGoal*featErr(i,:).').';
    % Compute corrective control
    uErr(i,:) = vel(i,:) + lambda_*poseErr(i,:);
end
    
%% Normalize data and fit GMM
Data = [poseErr'; uErr'];
in = 1:6;
out = 7:12; 
nbStates = 11; % GMM components
    
[Priors, Mu, Sigma] = EM_init_kmeans(Data, nbStates);
[Priors, Mu, Sigma] = EM(Data, Priors, Mu, Sigma);   

uRdsHandle = @(x) GMR(Priors, Mu, Sigma, x, in, out);

%% Simulate
demoLen = allDemoLen/length(demoNum);
sStar = featStar.';% Goal in feature space
Tcam = eye(4); % Initial pose of the dummy camera
figure;    
for d=demoNum
    pRds = zeros(3, demoLen);
    pRds(:,1) = poseInitAll(d,1:3)';
    vRds = zeros(6, demoLen);
    uRds  = zeros(6,demoLen-1);
    % Update camera position (orientation is fixed)   
    Tcam(1:3,4) = [pRds(1,1), pRds(2,1), pRds(3,1)];
        
    error = zeros(demoLen-1, 8);
    sRds = zeros(8, demoLen-1);
    for i=1:demoLen-1 % Run a bit longer to test for convergence
        % Project to image plane
        sCurr = cameraPoseToImagePoints(Tcam, P, KP);
        % Store features for plotting
        sRds(:,i) = reshape(sCurr, 8, 1);
        % Compute image error
        error(i,:) = reshape(sCurr, 8, 1) - sStar;
        % Compute RDS reshaping term
        uRds(:,i) = uRdsHandle(LpGoal*error(i,:).');
        % The corrective term in RDS should vanish to retrieve stability. 
        % Here we do it discontinuously. Use a smooth function in real cases 
        if i>=demoLen-1
            uRds(:,i) = zeros(6, 1);
        end
        % Compute RDS velocity
        e_ = error(i,:).';
        vRds(:,i) = -lambda_* LpGoal *e_ + uRds(:,i);
            
        % Update camera pose
        Tcam(1:3,4) = Tcam(1:3,4) + vRds(1:3,i)*dt_;
        pRds(:, i+1) =Tcam(1:3,4);
    end
        
    %% Plot results
    colorGreen = [0 127 0]/255;
    
    h1 = subplot(1,2,1);
    title('Cartesian position')
    plot(pRds(1,:), pRds(2,:), 'color', colorGreen, 'LineWidth', 3)
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
 
    plot(sRds(1,:), sRds(2,:), 'color', colorGreen, 'LineWidth', 3)
    plot(sRds(3,:), sRds(4,:), 'color', colorGreen, 'LineWidth', 3)
    plot(sRds(5,:), sRds(6,:), 'color', colorGreen, 'LineWidth', 3)
    plot(sRds(7,:), sRds(8,:), 'color', colorGreen, 'LineWidth', 3)

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