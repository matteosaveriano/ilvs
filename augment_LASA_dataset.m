%% 
% Compute a feature trajectory augmenting the LASA HandWriting Dataset
% The LASA dataset is available at: 
% https://bitbucket.org/khansari/lasahandwritingdataset/src/master/
%
% This function uses the Machine Vision Toolbox v4.2.1 from P. Corke
% See: https://petercorke.com/toolboxes/machine-vision-toolbox/

clear;
close all;

addpath(genpath('LASA_hand_writing_dataset/'));

modelPath = 'LASA_hand_writing_dataset/';

%% Visual servoing setup
% Create a default camera (see 'CentralCamera' documentation)
cam = CentralCamera('default');
% Create a dummy grid of 4 points (depth must be ~= 0)
P = mkgrid(2, 0.5, 'pose', SE3(0, 0, 5));

depth_goal = [5, 5, 5, 5]; % Final depth

%% Preprocess demonstrations
demoNum = 1:7;
dt = 0.0025;
PLOT = 1;
Tcam = cam.T.double;
posGoal = [1, 1, 1]; % Goal position (arbitrary)
oriGoal = cam.T.torpy'; % Orientation is fixed
for i=4:30
    % Load demonstrations
    [demos, ~, name] = load_LASA_models(modelPath, i);
    
    demoVS = [];
    for demoIt=demoNum   
        numData = size(demos{demoIt}.pos, 2);
        % Conver to meters and shift to goal
        demoVS{demoIt}.pos(1:2,:) = demos{demoIt}.pos./100 + repmat(posGoal(1:2)', 1, numData);
        % Add third dimension (linear motion)
        demoVS{demoIt}.pos(3,:) = linspace(Tcam(3,4), posGoal(3), numData);
        % Fixed orientation
        demoVS{demoIt}.pos(4:6,:) = repmat(oriGoal, 1, numData);
        % Compute Cartesian veloctiy
        demoVS{demoIt}.vel = [diff(demoVS{demoIt}.pos,[],2)./dt zeros(6,1)];
        
        % Project to image plane
        for n=1:numData
            cam.T = SE3(demoVS{demoIt}.pos(1,n), demoVS{demoIt}.pos(2,n), demoVS{demoIt}.pos(3,n));
            
            sCurr = cam.project(P);
            
            demoVS{demoIt}.feat(:,n) = reshape(sCurr, 8, 1);
        end
        % Store the sampling time
        demoVS{demoIt}.dt = dt;        
        
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
    
    filename = ['LASA_HandWriting_VS/DataSet/' name '_VS.mat'];
    save(filename, 'demoVS')
end
