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


%% Normalize data and fit GMM
scale_ = 1;
featErr = featErr./scale_;
Data = [featErr'; vel'];
in = 1:8;
out = 9:14; 
nbStates = 11; % GMM components
    
[Priors, Mu, Sigma] = EM_init_kmeans(Data, nbStates);
[Priors, Mu, Sigma] = EM(Data, Priors, Mu, Sigma);   

funHandle = @(x) GMR(Priors, Mu, Sigma, x, in, out);

lambda_ = 2; % Classic visual servoing gain

% Switching rule parameters
e1 = 50;
e0 = 5;
l1 = 1;
lo = 0;
l0 = 0;

%% Simulate
demoLen = allDemoLen/length(demoNum);
sStar = featStar.';% Goal in feature space
Tcam = eye(4); % Initial pose of the dummy camera
figure;    
nsVel = [0.5, 0, 0, 0, 0, 0]'; %0.1*rand(6,1); % Secondary task
for d=demoNum
    pRds = zeros(3, demoLen);
    pRds(:,1) = poseInitAll(d,1:3)';
    vRds = zeros(6, demoLen);
    % Update camera position (orientation is fixed)   
    Tcam(1:3,4) = [pRds(1,1), pRds(2,1), pRds(3,1)];
        
    error = zeros(demoLen-1, 8);
    nErr = sqrt(dot(featErr',featErr'));
    for i=1:200 %1:demoLen-1 % Run a bit longer to test for convergence
        % Project to image plane
        [sCurr, dCurr] = cameraPoseToImagePoints(Tcam, P, KP);
        % Compute image error
        error(i,:) = reshape(sCurr, 8, 1) - sStar;
        % Store features for plotting
        sRds(:,i) = reshape(sCurr, 8, 1);
        % Compute image Jacobian at goal
        e_ = error(i,:)';
        Ls = visualJacobianMatrix(sCurr, dCurr, KP); 
        %LvPinv =  norm(e_)*Ls'*e_/(e_'*(Ls*Ls')*e_);   
        
        ni = norm(e_); %norm(e_) - scale_*nErr(i);
        Lv = (e_'*Ls)./norm(e_);
        LvPinv = pinv(Lv);
        
        % Compute switching rule
        lambdaBar = switchingRule(norm(e_), e0, e1, l0, l1);
        
        %lambdaBar = 1;
        
        % Define primary task
        vNormErr = -lambda_* ni * LvPinv;
        vErr = -lambda_ * pinv(Ls) * e_;
        
        task1 = lambdaBar*vNormErr + (1-lambdaBar)*vErr;
        
        % Compute null-space projectors
        %Pv = eye(6) - (Ls'*(e_*e_')*Ls) / (e_'*(Ls*Ls')*e_);
        Pv = eye(6) - LvPinv*Lv;
        
        Ps = eye(6) - pinv(Ls)*Ls;
        
        % Define secondary task
        Pl = lambdaBar*Pv; % + (1-lambdaBar)*Ps;
        %task2 = - Pl * [Tcam(1:3,4) - 0.8*[1;1;1]; 0; 0; 0]; %nsVel;
        
      %  v2 = - [Tcam(1,4) - 1.2; 0; 0; 0; 0; 0];
        %if i<101
            v2 = funHandle(e_./scale_); %vel(i,:)';
        %end
        
        vDes(:,i) = v2;
        
        r2 = Ls*v2;
        task2 = pinv(Ls*Pl)*(r2 - Ls*vNormErr);
        
        task2 = Pl * v2;
        
        vRds(:,i) = task1 + task2;
        
        t1(:,i) = task1;
        t2(:,i) = task2;
        
        lBar(i) = lambdaBar;
        
        errNorm(i) = norm(e_);
        LvPinvNorm(i) = norm(LvPinv);
            
        % Update camera pose
        Tcam(1:3,4)  = Tcam(1:3,4) + vRds(1:3,i)*dt_;
        pRds(:, i+1) = Tcam(1:3,4);
    end
        
    %% Plot results
    colorGreen = [0 127 0]/255;
    
    
    figure(1)
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
    
   
%     
%     plot(error,'k')
    
   % close all
%     
   
%     figure
%     plot(t1','b')
%     hold on
%     plot(t2','k')
    
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
    
        
    figure(2)
    subplot(5,1,1)
    hold on
    plot(lBar,'k');
    title('Switching function')
    
    subplot(5,1,2)
    plot(errNorm,'k')
    hold on
    plot(sqrt(dot(scale_*featErr',scale_*featErr')),'b')
    title('Norm error')
    subplot(5,1,3)
    plot(LvPinvNorm, 'k')
    title('Norm Jv')
    subplot(5,1,4)
    hold on;
    plot(vRds(1,:)', 'r')
    plot(vRds(2,:)', 'g')
    plot(vRds(3,:)', 'b')
    title('Commanded velocity')
    box on; grid on;
    subplot(5,1,5)   
    hold on;
    plot(pRds(1,:)', 'r')
    plot(pRds(2,:)', 'g')
    plot(pRds(3,:)', 'b')
    title('Camera position')
    box on; grid on;
    
    figure(3)
    plot(errNorm,'k')
    hold on
    ee = sqrt(dot(scale_*featErr',scale_*featErr'));
    plot(ee(1:100),'b')
    title('Norm error')
    
%     
%     set(h2, 'YDir','reverse');
%     ax = h2;
%     ax.XLim = ax.XLim + [-10, 10];
%     ax.YLim = ax.YLim + [-10, 10];
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

% return;
% 
% for i=1:allDemoLen
%     % Compute Lp*e for each point
%     poseErr(i,:) = (LpGoal*featErr(i,:).').';
%     % Compute corrective control
%     uErr(i,:) = vel(i,:) + lambda_*poseErr(i,:);
% end
%     
% %% Normalize data and fit GMM
% Data = [poseErr'; uErr'];
% in = 1:6;
% out = 7:12; 
% nbStates = 11; % GMM components
%     
% [Priors, Mu, Sigma] = EM_init_kmeans(Data, nbStates);
% [Priors, Mu, Sigma] = EM(Data, Priors, Mu, Sigma);   
% 
% uRdsHandle = @(x) GMR(Priors, Mu, Sigma, x, in, out);

% %% Simulate
% demoLen = allDemoLen/length(demoNum);
% sStar = featStar.';% Goal in feature space
% Tcam = eye(4); % Initial pose of the dummy camera
% figure;    
% for d=demoNum
%     pRds = zeros(3, demoLen);
%     pRds(:,1) = poseInitAll(d,1:3)';
%     vRds = zeros(6, demoLen);
%     uRds  = zeros(6,demoLen-1);
%     % Update camera position (orientation is fixed)   
%     Tcam(1:3,4) = [pRds(1,1), pRds(2,1), pRds(3,1)];
%         
%     error = zeros(demoLen-1, 8);
%     sRds = zeros(8, demoLen-1);
%     for i=1:demoLen-1 % Run a bit longer to test for convergence
%         % Project to image plane
%         sCurr = cameraPoseToImagePoints(Tcam, P, KP);
%         % Store features for plotting
%         sRds(:,i) = reshape(sCurr, 8, 1);
%         % Compute image error
%         error(i,:) = reshape(sCurr, 8, 1) - sStar;
%         % Compute RDS reshaping term
%         uRds(:,i) = uRdsHandle(LpGoal*error(i,:).');
%         % The corrective term in RDS should vanish to retrieve stability. 
%         % Here we do it discontinuously. Use a smooth function in real cases 
%         if i>=demoLen-1
%             uRds(:,i) = zeros(6, 1);
%         end
%         % Compute RDS velocity
%         e_ = error(i,:).';
%         vRds(:,i) = -lambda_* LpGoal *e_ + uRds(:,i);
%             
%         % Update camera pose
%          Tcam(1:3,4) = Tcam(1:3,4) + vRds(1:3,i)*dt_;
%         pRds(:, i+1) =Tcam(1:3,4);
%     end
%         
%     %% Plot results
%     colorGreen = [0 127 0]/255;
%     
%     h1 = subplot(1,2,1);
%     title('Cartesian position')
%     plot(pRds(1,:), pRds(2,:), 'color', colorGreen, 'LineWidth', 3)
%     hold on
%     
%     plot(cameraPose{d}(1,1), cameraPose{d}(2,1), 'ko', 'MarkerSize', 15, 'MarkerFaceColor', 'k')
%     plot(cameraPose{d}(1,:), cameraPose{d}(2,:), 'k--', 'LineWidth', 3)
%     plot(cameraPose{d}(1,end), cameraPose{d}(2,end), 'kx', 'MarkerSize', 15, 'LineWidth', 3)
%     
%     set(h1, 'YDir','reverse');
%     ax = h1;
%     ax.XLim = ax.XLim + [-.1, .1];
%     ax.YLim = ax.YLim + [-.1, .1];
% 
%     h2 = subplot(1,2,2);
%     box on
%     hold on
%     title('Features position')
%  
%     plot(sRds(1,:), sRds(2,:), 'color', colorGreen, 'LineWidth', 3)
%     plot(sRds(3,:), sRds(4,:), 'color', colorGreen, 'LineWidth', 3)
%     plot(sRds(5,:), sRds(6,:), 'color', colorGreen, 'LineWidth', 3)
%     plot(sRds(7,:), sRds(8,:), 'color', colorGreen, 'LineWidth', 3)
% 
%     sDem = cameraFeat{d};
% 
%     plot(sDem(1,1), sDem(2,1), 'ko', 'MarkerSize', 15, 'MarkerFaceColor', 'k')
%     plot(sDem(3,1), sDem(4,1), 'ko', 'MarkerSize', 15, 'MarkerFaceColor', 'k')
%     plot(sDem(5,1), sDem(6,1), 'ko', 'MarkerSize', 15, 'MarkerFaceColor', 'k')
%     plot(sDem(7,1), sDem(8,1), 'ko', 'MarkerSize', 15, 'MarkerFaceColor', 'k')
%     
%     plot(sDem(1,:), sDem(2,:), 'k--', 'LineWidth', 3)
%     plot(sDem(3,:), sDem(4,:), 'k--', 'LineWidth', 3)
%     plot(sDem(5,:), sDem(6,:), 'k--', 'LineWidth', 3)
%     plot(sDem(7,:), sDem(8,:), 'k--', 'LineWidth', 3)
% 
%     plot(sStar(1,:), sStar(2,:), 'kx', 'MarkerSize', 15, 'LineWidth', 3)
%     plot(sStar(3,:), sStar(4,:), 'kx', 'MarkerSize', 15, 'LineWidth', 3)
%     plot(sStar(5,:), sStar(6,:), 'kx', 'MarkerSize', 15, 'LineWidth', 3)
%     plot(sStar(7,:), sStar(8,:), 'kx', 'MarkerSize', 15, 'LineWidth', 3)
%     
%     set(h2, 'YDir','reverse');
%     ax = h2;
%     ax.XLim = ax.XLim + [-10, 10];
%     ax.YLim = ax.YLim + [-10, 10];
% end