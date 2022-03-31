clear 

%% SEDSII initial CLF
dSEDSinit = [];
load('../Results/CLF_DELTA_DS/AllDist_no_retrain_27.mat');
dSEDSinit = [dSEDSinit allBestDist];
clear allBestDist;
load('../Results/CLF_DELTA_DS/AllDist_no_retrain_28.mat');
dSEDSinit = [dSEDSinit allBestDist];
clear allBestDist;
load('../Results/CLF_DELTA_DS/AllDist_no_retrain_29.mat');
dSEDSinit = [dSEDSinit allBestDist];
clear allBestDist;
load('../Results/CLF_DELTA_DS/AllDist_no_retrain_30.mat');
dSEDSinit = [dSEDSinit allBestDist];
clear allBestDist;

%% SEDSII re-trained CLF
dSEDSret = [];
load('../Results/CLF_DELTA_DS/AllDist_retrain_27.mat');
dSEDSret = [dSEDSret allBestDist];
clear allBestDist;
load('../Results/CLF_DELTA_DS/AllDist_retrain_28.mat');
dSEDSret = [dSEDSret allBestDist];
clear allBestDist;
load('../Results/CLF_DELTA_DS/AllDist_retrain_29.mat');
dSEDSret = [dSEDSret allBestDist];
clear allBestDist;
load('../Results/CLF_DELTA_DS/AllDist_retrain_30.mat');
dSEDSret = [dSEDSret allBestDist];
clear allBestDist;

%% SEDSII + R-DS
dRDS = [];
load('../Results/CLF_DELTA_DS/AllDist_clf_rds_27.mat');
dRDS = [dRDS allBestDist];
clear allBestDist;
load('../Results/CLF_DELTA_DS/AllDist_clf_rds_28.mat');
dRDS = [dRDS allBestDist];
clear allBestDist;
load('../Results/CLF_DELTA_DS/AllDist_clf_rds_29.mat');
dRDS = [dRDS allBestDist];
clear allBestDist;
load('../Results/CLF_DELTA_DS/AllDist_clf_rds_30.mat');
dRDS = [dRDS allBestDist];
clear allBestDist;

%% Compute median and quantiles
m_S_i    = median(dSEDSinit)
q10_S_i  = quantile(dSEDSinit, 0.1)
q90_S_i  = quantile(dSEDSinit, 0.9)


m_S_r    = median(dSEDSret)
q10_S_r  = quantile(dSEDSret, 0.1)
q90_S_r  = quantile(dSEDSret, 0.9)

m_RDS    = median(dRDS)
q10_RDS  = quantile(dRDS, 0.1)
q90_RDS  = quantile(dRDS, 0.9)

%% Compute mean and std
mu_S_i    = mean(dSEDSinit)
s_S_i    = std(dSEDSinit)

mu_S_r    = mean(dSEDSret)
s_S_r    = std(dSEDSret)

mu_RDS    = mean(dRDS)
s_RDS    = std(dRDS)