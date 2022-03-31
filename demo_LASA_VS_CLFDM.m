%% 
% Run CLF-DM approach on the LASA Visual Servoing HandWriting Dataset
%
% The original code for CLF-DM is available at: 
% https://bitbucket.org/khansari/clfdm/src/master/
%
% This function uses the Machine Vision Toolbox v4.2.1 from P. Corke
% See: https://petercorke.com/toolboxes/machine-vision-toolbox/

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
% 'doc preprocess_demos' in the MATLAB command window to get detailed
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

% Compute image Jacobian at goal
Lgoal = cam.visjac_p(reshape(featStar, 2, 4), depth_goal);
LpGoal = pinv(Lgoal);

for i=1:demoLen
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
    
u_clf_handle= @(y) DS_stabilizer(y, gmrHandle, Vxf, rho0, kappa0);

%% Simulate
demo_len = demoLen/length(demoNum); % Length of the single demonstration
s_star = featStar.'; % Goal in feature space
figure;
for d=demoNum
    p_clfdm = zeros(3, demo_len);
    p_clfdm(:,1) = poseInitAll(d,1:3)';
    v_clfdm = [0, 0, 0, 0, 0, 0]';
    u_clfdm = zeros(6,demo_len-1);

    cam.T = SE3(p_clfdm(1,1), p_clfdm(2,1), p_clfdm(3,1));
    error = zeros(demo_len-1, 8);
    s_clfdm = zeros(8, demo_len-1);    
    for i=1:demo_len-1
        s_curr = cam.project(P);

        % Store features for plotting
        s_clfdm(:,i) = reshape(s_curr, 8, 1);

        error(i,:) = reshape(s_curr, 8, 1) - s_star;

        u_clfdm(:,i) = u_clf_handle(LpGoal*error(i,:)');

        v_clfdm(:,i) = u_clfdm(:,i);

        % Update camera pose
        cam.T = cam.T.increment(v_clfdm(:,i)*dt_);
        p_clfdm(:, i+1) = cam.T.t;
    end
    v_clfdm(:,end+1) = [0, 0, 0, 0, 0, 0]';

    %% Plot results
    subplot(1,2,1)
    title('Cartesian position')
    plot(p_clfdm(1,:), p_clfdm(2,:), 'b')
    hold on
    
    plot(cameraPose{d}(1,1), cameraPose{d}(2,1), 'ko')
    plot(cameraPose{d}(1,:), cameraPose{d}(2,:), 'k')
    plot(cameraPose{d}(1,end), cameraPose{d}(2,end), 'kx')

    subplot(1,2,2)
    box on
    hold on
    title('Features position')
 
    plot(s_clfdm(1,:), s_clfdm(2,:), 'b')
    plot(s_clfdm(3,:), s_clfdm(4,:), 'b')
    plot(s_clfdm(5,:), s_clfdm(6,:), 'b')
    plot(s_clfdm(7,:), s_clfdm(8,:), 'b')

    s_dem = cameraFeat{d};

    plot(s_dem(1,1), s_dem(2,1), 'ko')
    plot(s_dem(3,1), s_dem(4,1), 'ko')
    plot(s_dem(5,1), s_dem(6,1), 'ko')
    plot(s_dem(7,1), s_dem(8,1), 'ko')
    
    plot(s_dem(1,:), s_dem(2,:), 'k')
    plot(s_dem(3,:), s_dem(4,:), 'k')
    plot(s_dem(5,:), s_dem(6,:), 'k')
    plot(s_dem(7,:), s_dem(8,:), 'k')

    plot(s_star(1,:), s_star(2,:), 'kx')
    plot(s_star(3,:), s_star(4,:), 'kx')
    plot(s_star(5,:), s_star(6,:), 'kx')
    plot(s_star(7,:), s_star(8,:), 'kx')
end