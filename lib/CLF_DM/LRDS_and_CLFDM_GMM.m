% This is a matlab script illustrating how to use CLFDM_lib to learn
% an arbitrary model from a set of demonstrations.

%Reference paper:
% S.M. Khansari-Zadeh and A. Billard (2014), "Learning Control Lyapunov Function to Ensure Stability 
% of Dynamical System-based Robot Reaching Motions." Robotics and Autonomous Systems, vol. 62, num 6, p. 752-765.

%%

close all
clear

addpath(genpath('/home/skiathos/Desktop/Incremental_Learning_DMP/code/Delta_GAS_DS'));
addpath(genpath('/home/skiathos/Desktop/Incremental_Learning_DMP/code/DS_Handles'));
addpath(genpath('/home/skiathos/Desktop/Incremental_Learning_DMP/code/DTW'));

addpath(genpath('/home/skiathos/Desktop/Incremental_Learning_DMP/code/LASA_hand_writing_dataset/'));
modelPath = '/home/skiathos/Desktop/Incremental_Learning_DMP/code/LASA_hand_writing_dataset/';

%% Putting CLFDM and GMR library in the MATLAB Path
if isempty(regexp(path,['CLFDM_lib' pathsep], 'once'))
    addpath([pwd, '/CLFDM_lib']);    % add SEDS dir to path
end
if isempty(regexp(path,['GMR_lib' pathsep], 'once'))
    addpath([pwd, '/GMR_lib']);    % add GMR dir to path
end

%% Load demonstrations
[demos, dt]   = load_LASA_models(modelPath, 27);
samplingRate  = 10;
simOptions.dt = dt*samplingRate;
demoSize = size(demos{1}.pos, 2);
sInd = 1:samplingRate:demoSize;
simOptions.Tmax  = 300;%length(sInd) - 20;

%% Load demonstrations
demoNum = [4 7 1];
gpResX  = [];
gpResXd = [];
x0_all  = [];
Data    = [];

for i=1:length(demoNum)   
    x0_all = [x0_all demos{demoNum(i)}.pos(:,1)];
    
    Data = [Data [demos{demoNum(i)}.pos(:,sInd); demos{demoNum(i)}.vel(:,sInd)]]; %finding initial points of all demonstrations
end

%% Train GMM
%Data = [Data [demos{demoNum(1)}.pos(:,sInd); demos{demoNum(1)}.vel(:,sInd)]];

[Priors0, Mu0, Sigma0] = EM_init_kmeans(Data, 4);
[Priors, Mu, Sigma] = EM(Data, Priors0, Mu0, Sigma0);

%% Define Reshaping GP
% GPR structures
% GPR(1)...GPR(N) desired state
GPR_reshape(1).logtheta = [log(3); log(sqrt(1.0)); log(sqrt(0.4))];
GPR_reshape(1).covfunc = {'covSum', {'covSEiso','covNoise'}};
GPR_reshape(1).data.in = [];
GPR_reshape(1).data.out = [];
GPR_reshape(2) = GPR_reshape(1);


for i=2:length(demoNum)
    gpResX  = [gpResX demos{demoNum(i)}.pos(:,sInd)];
    gpResXd = [gpResXd demos{demoNum(i)}.vel(:,sInd)];
end

% Create data matrix
%Data = [demos{demoNum(1)}.pos(:,sInd); demos{demoNum(1)}.vel(:,sInd)]; %finding initial points of all demonstrations


%% Setting parameters of the Control Lyapunov Function
clear Vxf0
modelNumber = 1;
switch modelNumber
    case 1
        Vxf0.L = 4; %the number of asymmetric quadratic components L>=0.
    case 2
        Vxf0.L = 1; %the number of asymmetric quadratic components L>=0.
end
Vxf0.d = size(demos{1}.pos, 1);
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

options.optimizePriors = true;% This is an added feature that is not reported in the paper. In fact
                              % the new CLFDM model now allows to add a prior weight to each quadratic
                              % energy term. IF optimizePriors sets to false, unifrom weight is considered;
                              % otherwise, it will be optimized by the sovler.
                              
options.upperBoundEigenValue = true; %This is also another added feature that is impelemnted recently.
                                     %When set to true, it forces the sum of eigenvalues of each P^l 
                                     %matrix to be equal one. 


tic;
%% Estimating an initial guess for the Lyapunov function
b_initRandom = false;
if b_initRandom
    lengthScale = sqrt(var(Data(1:Vxf0.d,:)'));
    lengthScaleMatrix = sqrtm(cov(Data(1:Vxf0.d,:)'));
    lengthScale = lengthScale(:);
    Vxf0.Priors = rand(Vxf0.L+1,1);
    for l=1:Vxf0.L+1
        tmpMat = randn(Vxf0.d,Vxf0.d);
        Vxf0.Mu(:,l) = randn(Vxf0.d,1).*lengthScale;
        Vxf0.P(:,:,l) = lengthScaleMatrix*(tmpMat*tmpMat')*lengthScaleMatrix;
    end
else
    Vxf0.Priors = ones(Vxf0.L+1,1);
    Vxf0.Priors = Vxf0.Priors/sum(Vxf0.Priors);
    Vxf0.Mu = zeros(Vxf0.d,Vxf0.L+1);
    for l=1:Vxf0.L+1
        Vxf0.P(:,:,l) = eye(Vxf0.d);
    end
end

% Solving the optimization
Vxf = learnEnergy(Vxf0,Data,options);
timeCLF = toc;

%% Simulation
% A set of options that will be passed to the Simulator. Please type 
% 'doc preprocess_demos' in the MATLAB command window to get detailed
% information about each option.
opt_sim.dt = simOptions.dt;
opt_sim.i_max = 4000;
opt_sim.tol = .3;
opt_sim.plot = false;
d = size(Data,1)/2; %dimension of data

% rho0 and kappa0 impose minimum acceptable rate of decrease in the energy
% function during the motion. Refer to page 8 of the paper for more information
rho0 = 1;
kappa0 = 0.1;
% 
in = 1:Vxf.d;
out = Vxf.d+1:2*Vxf.d;
fn_handle_GMR = @(x) GMR(Priors, Mu, Sigma, x, in,out);
fn_handle = @(y) DS_stabilizer(y,fn_handle_GMR,Vxf,rho0,kappa0);
% 
% %% Train Reshaping GP
for itNum=1:1
tic;
trainingData = computeTrainingData(gpResX, gpResXd, fn_handle, false);
 
GPR_reshape = trainGPIncremental(GPR_reshape, gpResX, trainingData, 3, true);
timeIncrGp = toc;

fn_handle_LRDS = @(y) MD_GP_Regression(GPR_reshape, y);

truncValues.alphaMin = 0.01;
truncValues.rho      = 0.01;

simOptions.iterNum    = 1000;
simOptions.tol        = .3;
simOptions.goal       = [0; 0];
simOptions.plotResult = false;
simOptions.Tmax  = 300; %length(sInd) - 30;% - 20;

x = reshapeCLF(fn_handle, fn_handle_LRDS , x0_all, simOptions);

%x = Simulation(x0_all,[],fn_handle,opt_sim); %running the simulator

%% Plotting the result
plotResult = true;
if(plotResult)
    fig = figure;
    sp = gca;
    hold on
    h(1) = plot(sp,demos{demoNum(1)}.pos(1,sInd),demos{demoNum(1)}.pos(2,sInd),'o','color',[0, 180/255, 0],'LineWidth',2,'Markersize',10);
    h(2) = plot(sp,GPR_reshape(1).data.in(1,1:15),GPR_reshape(1).data.in(2,1:15),'o','color',[183, 65, 14]./255,'LineWidth',2,'Markersize',10);
    h(2) = plot(sp,GPR_reshape(1).data.in(1,16:end),GPR_reshape(1).data.in(2,16:end),'o','color',[200/255, 0, 0],'LineWidth',2,'Markersize',10);
%     h(2) = plot(sp,GPR_reshape(1).data.in(1,18:end),GPR_reshape(1).data.in(2,18:end),'o','color',[200/255, 0, 0],'LineWidth',2,'Markersize',10);
    axis tight
    ax=get(gca);
%     axis([ax.XLim(1)-(ax.XLim(2)-ax.XLim(1))/10 ax.XLim(2)+(ax.XLim(2)-ax.XLim(1))/10 ...
%           ax.YLim(1)-(ax.YLim(2)-ax.YLim(1))/10 ax.YLim(2)+(ax.YLim(2)-ax.YLim(1))/10]);


    %h(4) = EnergyContour(Vxf,axis,[],[],sp, [], false);

    xlabel('x (mm)','fontsize',15);
    ylabel('y (mm)','fontsize',15);
    
%     D = [ax.XLim(1)-(ax.XLim(2)-ax.XLim(1))/10 ax.XLim(2)+(ax.XLim(2)-ax.XLim(1))/10 ...
%           ax.YLim(1)-(ax.YLim(2)-ax.YLim(1))/10 ax.YLim(2)+(ax.YLim(2)-ax.YLim(1))/10];

    D = [ax.XLim(1)-(ax.XLim(2)-ax.XLim(1))/10 27+(ax.XLim(2)-ax.XLim(1))/10 ...
          ax.YLim(1)-(ax.YLim(2)-ax.YLim(1))/10 ax.YLim(2)+(ax.YLim(2)-ax.YLim(1))/10];
    plotDSStreamLines(fn_handle, fn_handle_LRDS, D)

    for i=1:length(x)
       h(5) = plot(sp,x{i}(1,1:end-1),x{i}(2,1:end-1),'k','linewidth',4);
    end
    h(3) = plot(0,0,'k.','markersize',50,'linewidth',2);
    %lg = legend(h,'demonstrations','target','energy levels','reproductions','location','southwest','orientation','horizontal');
    %set(lg,'position',[0.0673    0.9278    0.8768    0.0571])
end

% totTime(1,itNum) = timeGP;
% totTime(2,itNum) = timeCLF;

totTime(1,itNum) = timeIncrGp;
end





