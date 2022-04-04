%% 
% Compute a feature trajectory augmenting the LASA HandWriting Dataset
%
% The LASA dataset is available at: 
% https://bitbucket.org/khansari/lasahandwritingdataset/src/master/
%

%%
clear;
close all;

addpath(genpath('datasets/'));
addpath(genpath('lib/'));

modelPath = 'LASA_hand_writing_dataset/';

%% Visual servoing setup
% Create a dummy grid of 4 points
P = [-0.2500, -0.2500, 0.2500,  0.2500; ...
     -0.2500,  0.2500, 0.2500, -0.2500; ...
      5.0000,  5.0000, 5.0000,  5.0000 ];
% Create a dummy pinhole camera
% Camera matrix
KP = zeros(3,4);  
KP(1,1) = 800;
KP(2,2) = 800;
KP(3,3) = 1;
KP(1:2,3) = 512;

depth_goal = [5, 5, 5, 5]; % Final depth

%% Preprocess demonstrations
demoNum = 1:7; % Consider all LASA demonstrations
dt = 0.0025; % Sampling time
PLOT = 1; % Plot results
Tcam = eye(4); % Initial pose of the dummy camera
posGoal = ones(3,1); % Goal position (arbitrary)
oriGoal = zeros(3,1); % Orientation is fixed
for i=4:30
    % Load demonstrations
    [demos, ~, name] = load_LASA_models(modelPath, i);
    
    demoVS = [];
    for demoIt=demoNum   
        numData = size(demos{demoIt}.pos, 2);
        % Conver to meters and shift to goal
        demoVS{demoIt}.pos(1:2,:) = demos{demoIt}.pos./100 + repmat(posGoal(1:2,1), 1, numData);
        % Add third dimension (linear motion)
        demoVS{demoIt}.pos(3,:) = linspace(Tcam(3,4), posGoal(3), numData);
        % Fixed orientation
        demoVS{demoIt}.pos(4:6,:) = repmat(oriGoal, 1, numData);
        % Compute Cartesian veloctiy
        demoVS{demoIt}.vel = [diff(demoVS{demoIt}.pos,[],2)./dt zeros(6,1)];
        
        for n=1:numData
            % Update camera pose
            Tcam(1:3,4) = [demoVS{demoIt}.pos(1,n); demoVS{demoIt}.pos(2,n); demoVS{demoIt}.pos(3,n)];
            % Project to image plane
            sCurr = cameraPoseToImagePoints(Tcam, P, KP);
            % Store data   
            demoVS{demoIt}.feat(:,n) = reshape(sCurr, 8, 1);
        end
        % Store the sampling time
        demoVS{demoIt}.dt = dt;        
        
		% Plot generated data
        if(PLOT)
            figure(i)
            subplot(1,2,1)
            plot(demoVS{demoIt}.pos(1,:), demoVS{demoIt}.pos(2,:), 'k')
            hold on

            subplot(1,2,2)
            hold on
            s_dem = demoVS{demoIt}.feat;

            plot(s_dem(1,:), s_dem(2,:), 'k')
            plot(s_dem(3,:), s_dem(4,:), 'r')
            plot(s_dem(5,:), s_dem(6,:), 'b')
            plot(s_dem(7,:), s_dem(8,:), 'm')
        end
        
        % Add point grid and depth
        demoVS{demoIt}.point_grid = P;
        demoVS{demoIt}.depth_goal = depth_goal;
    end
    
    filename = ['datasets/LASA_HandWriting_VS/DataSet/' name '_VS.mat'];
    save(filename, 'demoVS')
end
