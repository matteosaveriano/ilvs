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

% for i=1:4
%    sGoal(:,i) = pixel2normalized(cam, fGoal(:,i));
% end
sGoal = fGoal;


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
normFeatErr = [];
dNormFeatErr = [];
gmmIn = [];
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
    
    fE = cameraFeat{i}' - repmat(cameraFeat{i}(:,end), 1, length(cameraFeat{i}))';
    nFe = sqrt(dot(fE', fE'));
    
    normFeatErr = [normFeatErr, nFe];
    dNormFeatErr = [dNormFeatErr, gradient(nFe)./dt_];
    gmmIn = [gmmIn, linspace(0,1,length(cameraFeat{i}))];
end

allDemoLen = size(feat, 1);
featStar = mean(featStarAll);
poseStar = mean(poseStarAll);

featErr = feat - repmat(featStar, allDemoLen, 1);
poseErr = zeros(allDemoLen, 6);
uErr = zeros(allDemoLen, 6);

%% Normalize data and fit GMM
Data = [featErr'; vel'];
in = 1:8;
out = 9:14;
nbStates = 11; % GMM components

[Priors, Mu, Sigma] = EM_init_kmeans(Data, nbStates);
[Priors, Mu, Sigma] = EM(Data, Priors, Mu, Sigma);

funHandle = @(x) GMR(Priors, Mu, Sigma, x, in, out);

Data = [gmmIn; normFeatErr; dNormFeatErr];
in = 1;
out = 2:3;
nbStates = 7; % GMM components

[Priors, Mu, Sigma] = EM_init_kmeans(Data, nbStates);
[Priors, Mu, Sigma] = EM(Data, Priors, Mu, Sigma);

funHandleNorm = @(x) GMR(Priors, Mu, Sigma, x, in, out);

% Switching rule parameters
e1 = 40;
e0 = 20;
l1 = 1;
l0 = 0;

lambda = 1;
dt = dt_; %1/30;
L = 200;
%W = eye(8);

for d=1
    pInit = poseInitAll(d,1:3)' + 0.1*rand(1,3);
    cam.T = SE3(pInit(1), pInit(2), pInit(3));
    for i = 1:L
        % Project to image plane
        fCurr = cam.project(P);
        %     for it=1:4
        %         sCurr(:,it) = pixel2normalized(cam, fCurr(:,it));
        %     end
        sCurr = fCurr;
        % Compute error
        sRds(:,i) = reshape(fCurr, 8, 1);
        
        error(:,i) = reshape(sCurr - sGoal, 8, 1);
        e_ = error(:,i);
        
        normE = norm(e_);
        nError(i) = normE;
        
        
        % Compute interaction matrix
        
        Ls = cam.visjac_p(fCurr, depthGoal);
        %     Ls(1:2,:) = (1/cam.f) * diag(cam.rho) * Lf(1:2,:);
        %     Ls(3:4,:) = (1/cam.f) * diag(cam.rho) * Lf(3:4,:);
        %     Ls(5:6,:) = (1/cam.f) * diag(cam.rho) * Lf(5:6,:);
        %     Ls(7:8,:) = (1/cam.f) * diag(cam.rho) * Lf(7:8,:);
        
        % Compute jacobian and projector
        [Lv, LvPinv, Pv, den(i)] = computeLvPinv(e_, Ls);
        
        % Compute switching rule
        lBar1 = switchingRule(norm(e_), e0, e1, l0, l1);
        % Apparamm' a Chomett
        lBar2 = switchingRule(norm(e_'*Ls), e0, e1, l0, l1);
        
        lBar = lBar1 * lBar2;
        
        lambdaBar(i) = lBar;
        
        
        % Compute first task
        etaD(:,i) = funHandleNorm(2*i/L);
        task1_1 =  LvPinv * (-lambda * (normE - etaD(1,i)) + etaD(2,i));
        task1_2 = -lambda * pinv(Ls) * e_;
        task1 = lBar*task1_1 + (1-lBar)*task1_2;
        
        % Compute secondary task
        r = funHandle(e_);
        
        rVec(:,i) = r;
        
        task2_1 = Pv * r;
        % task2_2 = (eye(6) - pinv(Ls)*Ls) * r;
        
        %rd2 = Ls*r;
        %task2_1 = pinv(Ls*Pv)*(rd2 - Ls*task1_1);
        
        %Ps = (eye(6) - pinv(Ls)*Ls);
        %task2_2 = pinv(Ls*Ps)*(rd2 - Ls*task1_2);
        
        task2 = lBar*task2_1; % + (1-lBar)*task2_2;
        
        vCam(:,i) = task1 + task2;
        vCam(4:6,i) = 0;
        
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
    hold on;
    plot(nError, 'r')
    plot(etaD(1,:),'k')
    title('Error norm')
    subplot(7,1,3)
    plot(LvP(1:end))
    title('LvPinv norm')
    subplot(7,1,4)
    hold on;
    % plot(vCam(1,:),'r')
    % plot(vCam(2,:),'g')
    % plot(vCam(3,:),'b')
    plot(vCam(1:3,:)','b')
    plot(rVec(1:3,:)','k--')
    plot(vel(1:100,1:3),'m-.')
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
    
    
    figure(2)
    colorGreen = [0 127 0]/255;
    h1 = subplot(1,2,1);
    title('Cartesian position')
    hold on
    plot(pCam(1,:), pCam(2,:), 'color', colorGreen, 'LineWidth', 3)
    
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
    
    fGoal = fGoal(:);
    plot(fGoal(1,:), fGoal(2,:), 'kx', 'MarkerSize', 15, 'LineWidth', 3)
    plot(fGoal(3,:), fGoal(4,:), 'kx', 'MarkerSize', 15, 'LineWidth', 3)
    plot(fGoal(5,:), fGoal(6,:), 'kx', 'MarkerSize', 15, 'LineWidth', 3)
    plot(fGoal(7,:), fGoal(8,:), 'kx', 'MarkerSize', 15, 'LineWidth', 3)
    
end


function [Lv, LvPinv, Pv, den] = computeLvPinv(e, Ls)
normE = norm(e);
den = e' * (Ls * Ls') * e;

Lv = (e' * Ls) ./ normE;

LvPinv = (normE * Ls' * e) / den;
Pv = eye(6) - (Ls'*(e*e')*Ls)/den;
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


function xy = pixel2normalized(cam, uv)
xy(1,1) = (uv(1) - cam.u0) * cam.rho(1) / cam.f;
xy(2,1) = (uv(2) - cam.v0) * cam.rho(2) / cam.f;
end