% README
% 
% Additional data of the Silverbox system, not described in the technical
% note (http://www.it.uu.se/research/publications/reports/2013-006/2013-006-nc.pdf)
% can be found in Schroeder80mV.mat or Schroeder80mV.csv. This dataset
% contains, amongst others, a Schroeder phase multisine measurement. This
% one period of the schroeder phase multisine can be extracted with the
% following (Matlab code). The measurement settings are the same as for the
% data described in the technical note.

load('Schroeder80mV.mat')

V1=V1-mean(V1); % Remove offset errors on the input measurements (these are visible in the zero sections of the input)
                % The input is designed to have zero mean
V2=V2-mean(V2); % Approximately remove the offset errors on the output measurements. 
                % This is an approximation because the silverbox can create itself also a small DC level 
 
uSchroeder=V1(10585:10585+1023);  % select the Schroeder section of the experiment
ySchroeder=V2(10585:10585+1023);
% One period is 1024 points. Only the odd frequencies bins (f0,3f0,5f0,...) 
% are excited. f0 = fs/N, N=1024.