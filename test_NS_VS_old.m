clear;
close all;

addpath(genpath('lib'))
addpath(genpath('datasets'))


%% Create a dummy pinhole camera
cam = CentralCamera('default');

depthGoal = [4, 4, 4, 4];

P = [-0.2500   -0.2500    0.2500    0.2500; ...
     -0.2500    0.2500    0.2500   -0.2500; ...
      5.0000    5.0000    5.0000    5.0000 ];

fGoal = [262 262 262 362 362 362 362 262]';

fGoal = reshape(fGoal, 2, 4);

for i=1:4
   sGoal(:,i) = pixel2normalized(cam, fGoal(:,i)); 
end

%% Learning and reproduction loop
motionClass = 3; % Which of the 30 motions to use
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

pInit = poseInitAll(1,1:3)';
cam.T = SE3(pInit(1), pInit(2), pInit(3));


%% Normalize data and fit GMM
Data = [featErr'; vel'];
in = 1:8;
out = 9:14; 
nbStates = 11; % GMM components
    
[Priors, Mu, Sigma] = EM_init_kmeans(Data, nbStates);
[Priors, Mu, Sigma] = EM(Data, Priors, Mu, Sigma);   

funHandle = @(x) GMR(Priors, Mu, Sigma, x, in, out);

% Switching rule parameters
e1 = 0.05;
e0 = 0.02;
l1 = 1;
l0 = 0;

lambda = 1;
dt = 1/30;
L = 200;

for i = 1:L
    r = [0.1*sin((i-1)*dt_), 0, 0, 0, 0, 0]';
    % Project to image plane
    fCurr = cam.project(P);
    for it=1:4
        sCurr(:,it) = pixel2normalized(cam, fCurr(:,it)); 
    end
    % Compute error
    error(:,i) = reshape(sCurr - sGoal, 8, 1);
    nError(i) = norm(error(:,i));
    
    % Compute interaction matrix
    Lf = cam.visjac_p(fCurr, depthGoal); 
    Ls(1:2,:) = (1/cam.f) * diag(cam.rho) * Lf(1:2,:);
    Ls(3:4,:) = (1/cam.f) * diag(cam.rho) * Lf(3:4,:);
    Ls(5:6,:) = (1/cam.f) * diag(cam.rho) * Lf(5:6,:);
    Ls(7:8,:) = (1/cam.f) * diag(cam.rho) * Lf(7:8,:);
    
    % Compute jacobian and projector 
    e_ = error(:,i);
    [Lv, LvPinv, Pv, den(i)] = computeLvPinv(e_, Ls);
    % Compute switching rule
    lBar = switchingRule(norm(e_), e0, e1, l0, l1);
    
    lambdaBar(i) = lBar;
        
    % Compute first task
    task1_1 = -lambda * LvPinv * norm(e_);
    task1_2 = -lambda * pinv(Ls) * e_;
    task1 = lBar*task1_1 + (1-lBar)*task1_2;
    
    % Compute secondary task 
    
    ePixel = reshape(fCurr - fGoal, 8, 1);
    %r = funHandle(ePixel);
    
    task2_1 = Pv * r;
    % task2_2 = (eye(6) - pinv(Ls)*Ls) * r; 
    
    %rd2 = Ls*r;
    %task2_1 = pinv(Ls*Pv)*(rd2 - Ls*task1_1);
    
    %Ps = (eye(6) - pinv(Ls)*Ls);
    %task2_2 = pinv(Ls*Ps)*(rd2 - Ls*task1_2); 
    
    task2 = lBar*task2_1; % + (1-lBar)*task2_2;
    
    vCam(:,i) = task1 + task2; 
    % Integration
    cam.T = cam.T.increment(vCam(:,i)*dt);
    pCam(:,i) = cam.T.t;
    
    LvP(i) = norm(LvPinv);
    
    jacTask2(i) = Lv * Pv * r;
    
    LstE(i) = norm(Ls'*e_);
end

figure(1)
subplot(7,1,1)
hold on;
plot(pCam(1,:),'r')
plot(pCam(2,:),'g')
plot(pCam(3,:),'b')
title('Camera position')
subplot(7,1,2)
plot(nError(1:end))
title('Error norm')
subplot(7,1,3)
plot(LvP(1:end))
title('LvPinv norm')
subplot(7,1,4)
hold on;
plot(vCam(1,:),'r')
plot(vCam(2,:),'g')
plot(vCam(3,:),'b')
plot(vCam(4:6,:)','k--')
title('Camera velocity')
subplot(7,1,5)
plot(lambdaBar)
title('Switching')
subplot(7,1,6)
plot(jacTask2)
title('Jacobian times Task 2')
subplot(7,1,7)
plot(LstE)
title('Jacobian times E')


function [Lv, LvPinv, Pv, den] = computeLvPinv(e, Ls)
    normE = norm(e);
    den = e' * (Ls * Ls') * e;
    
    Lv = (e' * Ls) ./ normE;
    LvPinv = (normE * Ls' * e) / den;
    Pv = eye(6) - (Ls'*(e*e')*Ls)/den;
end

function xy = pixel2normalized(cam, uv)
    xy(1,1) = (uv(1) - cam.u0) * cam.rho(1) / cam.f;
    xy(2,1) = (uv(2) - cam.v0) * cam.rho(2) / cam.f;
end

function lambdaBar = switchingRule(errNorm, e0, e1, l0, l1)
    lambda = 1 / (1 + exp(-12*((errNorm - e0)/(e1 - e0))+6));
    lambdaBar = (lambda - l0)/(l1 - l0);
    if errNorm < e0
        lambdaBar = 0;
    elseif errNorm > e1
        lambdaBar = 1;
    end
end