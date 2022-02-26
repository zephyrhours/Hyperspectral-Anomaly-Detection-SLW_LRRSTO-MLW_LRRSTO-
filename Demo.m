%% ################################################################
% Paper£º
% "Anomaly Detection in Hyperspectral Imagery based on Low-Rank Representation incorporating Spatial Constrain"
% Compiled by ZephyrHou on 2018-06-10
% Revised by ZephyrHou on 2019-06-06
%% Main Function
clc;clear;close all;
addpath(genpath(pwd));
disp('Compiled by ZephyrHou')
disp(datestr(now))
%% Datasets
% load Salians_120_sj25_syn.mat;
% load Salians_120_sj25_gt.mat;

load HYDICE_80_100.mat;
load HYDICE_80_100_gt.mat;

% load AVIRIS_100_100.mat;
% load AVIRIS_100_100_gt58.mat;

% load Viareggio.mat
% load Viareggio_gt.mat

%% ###############################################################
%% Preprocessing
[rows,cols,bands]=size(hsi);
hsi = reshape(hsi,rows*cols,bands);
label_value = reshape(hsi_gt,1,rows*cols);

%% Normalization
hsi = hsi';
max_y = max(max(hsi));
min_x = min(min(hsi));
hsi = (hsi-min_x)/(max_y-min_x);

%% Random selection dictionary atoms(300 atoms)
dic_num = 300;
J = randperm(rows*cols);
J = J(1:dic_num);
A = hsi(:,J);
hsi = reshape(hsi',rows,cols,bands);

% ------- Import the Prepared dictionary -----------
% load('Dictionary_AVIRIS.mat')
% load('Dictionary_HYDICE.mat')
% load('Dictionary_Salinas.mat')
% load('Dictionary_Viareggio.mat')

%% Proposed: SLW_LRRSTO
beta = 1;
lambda = 0.1;
disp('Runing SLW_LRRSTO......')

% min |Z|_* + beta*||Z-ZV||_F^2 + lambda*|E|_2,1 s.t., X = AZ+E; I= [I1,I2...,I9]; ZV = [ZV1,...,ZV9];
tic
[SLW_LRRSTO_out] = func_SLW_LRRSTO( hsi,A,beta,lambda);
t1=toc;

SLW_LRRSTO_value = reshape(SLW_LRRSTO_out,1,rows*cols);
[FA_SLW_LRRSTO,PD_SLW_LRRSTO] = perfcurve(label_value,SLW_LRRSTO_value,'1') ;
AUC_SLW_LRRSTO=-sum((FA_SLW_LRRSTO(1:end-1)-FA_SLW_LRRSTO(2:end)).*(PD_SLW_LRRSTO(2:end)+PD_SLW_LRRSTO(1:end-1))/2);
disp(['AUC=',num2str(AUC_SLW_LRRSTO)]);
disp('SLW_LRRSTO Finished')

%% Proposed: MLW_LRRSTO
beta = 1;
lambda = 0.1;
disp('Runing MLW_LRRSTO......')

% min |Z|_* + beta*||I*Z-ZV||_F^2 + lambda*|E|_2,1 s.t., X = AZ+E; I= [I1,I2...,I9]; ZV = [ZV1,...,ZV9];
tic
MLW_LRRSTO_out=func_MLW_LRRSTO( hsi,A,beta,lambda); 
t2=toc;

MLW_LRRSTO_value = reshape(MLW_LRRSTO_out,1,rows*cols);
[FA_MLW_LRRSTO,PD_MLW_LRRSTO] = perfcurve(label_value,MLW_LRRSTO_value,'1') ;
AUC_MLW_LRRSTO=-sum((FA_MLW_LRRSTO(1:end-1)-FA_MLW_LRRSTO(2:end)).*(PD_MLW_LRRSTO(2:end)+PD_MLW_LRRSTO(1:end-1))/2);
disp(['AUC=',num2str(AUC_MLW_LRRSTO)]);
disp('MLW_LRRSTO Finished')

%% #################################################################
%% AUC Values and Exectution Time
clc;
disp('Compiled by ZephyrHou')
disp('------------------------ Results Display ---------------------------')
disp('SLW_LRRSTO:');
disp(['AUC=',num2str(AUC_SLW_LRRSTO)]);
disp(['SLW_LRRSTO time: ', num2str(t1)]);
disp('------------------------------')
disp('MLW_LRRSTO:');
disp(['AUC=',num2str(AUC_MLW_LRRSTO)]);
disp(['MLW_LRRSTO time: ', num2str(t2)]);
disp('-------------------------- The end --------------------------------')

%% ROC Curves
figure('Name','ROC Curves1')
plot(FA_SLW_LRRSTO, PD_SLW_LRRSTO, 'r-', 'LineWidth', 2);hold on;
plot(FA_MLW_LRRSTO, PD_MLW_LRRSTO, 'k-', 'LineWidth', 2);hold on;
legend('SLW\_LRRSTO','MLW\_LRRSTO')
title('ROC Curves');

figure('Name','ROC Curves2')
semilogx(FA_SLW_LRRSTO, PD_SLW_LRRSTO, 'r-', 'LineWidth', 2);hold on;
semilogx(FA_MLW_LRRSTO, PD_MLW_LRRSTO, 'k-', 'LineWidth', 2);hold on;
legend('SLW\_LRRSTO','MLW\_LRRSTO')
xlim([1e-4 1])
title('ROC Curves');

