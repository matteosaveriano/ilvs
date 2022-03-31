%% 
% Run RDS approach on the LASA Visual Servoing HandWriting Dataset
%
% This function uses the Machine Vision Toolbox v4.2.1 from P. Corke
% See: https://petercorke.com/toolboxes/machine-vision-toolbox/

%%
clear;
close all;

addpath(genpath('lib'))
addpath(genpath('datasets'))

%% Camera and feature setup
% Create a default camera (see 'CentralCamera' documentation)
cam = CentralCamera('default');
          
%% Learning and reproduction loop
motionClass = 5; % Which of the 30 motions to use
demoNum = 1:3; % Consider only the first 3 demonstrations
samplingRate = 10; % Resample the trajectory to make learning faster

%% Create training data
% Load demonstrations
[demos, name] = load_LASA_VS_models('LASA_HandWriting_VS/', motionClass);
demoSize = size(demos{1}.pos, 2);
sInd = [1:samplingRate:demoSize demoSize]; % Use only points at 'sInd'
    
% Retrieve point grid, goal depth, and sampling time
P = demos{1}.point_grid;
depth_goal = demos{1}.depth_goal; % Final depth
    

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
    
demoLen = size(feat, 1);
featStar = mean(featStarAll);
poseStar = mean(poseStarAll);
    
featErr = feat - repmat(featStar, demoLen, 1);
poseErr = zeros(demoLen, 6);
uErr = zeros(demoLen, 6);

lambda_ = 1; % Classic visual servoing gain

% Compute image Jacobian at goal
Lgoal = cam.visjac_p(reshape(featStar, 2, 4), depth_goal);
LpGoal = pinv(Lgoal);
for i=1:demoLen
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

u_rds_handle = @(x) GMR(Priors, Mu, Sigma, x, in, out);

%% Simulate
demo_len = demoLen/length(demoNum);
s_star = featStar.';% Goal in feature space
figure;    
for d=demoNum
    p_rds = zeros(3, demo_len);
    p_rds(:,1) = poseInitAll(d,1:3)';
    v_rds = [0, 0, 0, 0, 0, 0]';
    u_rds  = zeros(6,demo_len-1);
        
    cam.T = SE3(p_rds(1,1), p_rds(2,1), p_rds(3,1));
        
    error = zeros(demo_len-1, 8);
    s_rds = zeros(8, demo_len-1);
    for i=1:demo_len-1 % Run a bit longer to test for convergence
        s_curr = cam.project(P);
        % Store features for plotting
        s_rds(:,i) = reshape(s_curr, 8, 1);

        error(i,:) = reshape(s_curr, 8, 1) - s_star;

        u_rds(:,i) = u_rds_handle(LpGoal*error(i,:).');
        % The corrective term in RDS should vanish to retrieve stability. 
        % Here we do it discontinuously. Use a smooth function in real cases 
        if i>=demo_len-1
            u_rds(:,i) = zeros(6, 1);
        end

        e_ = error(i,:).';
        v_rds(:,i) = -lambda_* LpGoal *e_ + u_rds(:,i);
            
        % Update camera pose
        cam.T = cam.T.increment(v_rds(:,i)*dt_);
        p_rds(:, i+1) = cam.T.t;
    end
    v_rds(:,end+1) = [0, 0, 0, 0, 0, 0]';

        
        PLOT = 1;
        colorGreen = [0 127 0]/255;
        if(PLOT)
            %figure(modIt)
            h1 = figure(1);
            plot(p_rds(1,:), p_rds(2,:), 'color',colorGreen, 'LineWidth', 3)
            hold on
            plot(cameraPose{d}(1,:), cameraPose{d}(2,:), 'k--', 'LineWidth', 3)

            h2 = figure(2);
            %subplot(1,2,2)
            box on
            hold on
            
            plot(s_rds(1,:), s_rds(2,:), 'color',colorGreen, 'LineWidth', 3)
            plot(s_rds(3,:), s_rds(4,:), 'color',colorGreen, 'LineWidth', 3)
            plot(s_rds(5,:), s_rds(6,:), 'color',colorGreen, 'LineWidth', 3)
            plot(s_rds(7,:), s_rds(8,:), 'color',colorGreen, 'LineWidth', 3)
            
            s_dem = cameraFeat{d};

            plot(s_dem(1,:), s_dem(2,:), 'k--', 'LineWidth', 3)
            plot(s_dem(3,:), s_dem(4,:), 'k--', 'LineWidth', 3)
            plot(s_dem(5,:), s_dem(6,:), 'k--', 'LineWidth', 3)
            plot(s_dem(7,:), s_dem(8,:), 'k--', 'LineWidth', 3)
        end    
    end

    %subplot(1,2,1)
    plot(cameraPose{1}(2,end), cameraPose{1}(3,end), 'k.', 'MarkerSize', 30)
    %axis([0.5 1.5 0.5 1.5])
    axis square;
    set(gca, 'XTick', [], 'YTick', []);

    % Plot goals
    figure(2)
    %subplot(1,2,2)
    plot(s_star(1,:), s_star(2,:), 'k.', 'MarkerSize', 30)
    plot(s_star(3,:), s_star(4,:), 'k.', 'MarkerSize', 30)
    plot(s_star(5,:), s_star(6,:), 'k.', 'MarkerSize', 30)
    plot(s_star(7,:), s_star(8,:), 'k.', 'MarkerSize', 30) 
    %axis([200 500 200 500])
    axis square;
    set(gca, 'YDir','reverse', 'XTick', [], 'YTick', []);
    ax = gca;
    ax.XLim = ax.XLim + [-10, 10];
    ax.YLim = ax.YLim + [-10, 10];
end