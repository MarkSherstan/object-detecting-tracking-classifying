function [] = confusion_matrix(T,Y)
% Reorganize data so MATLAB can plot. See reference [5].

T = grp2idx(T)';
Y = grp2idx(Y)';

M = size(unique(T),2);
N = size(T,2);

targets = zeros(M,N);
outputs = zeros(M,N);

targetsIdx = sub2ind(size(targets), T, 1:N);
outputsIdx = sub2ind(size(outputs), Y, 1:N);

targets(targetsIdx) = 1;
outputs(outputsIdx) = 1;

plotconfusion(targets,outputs)
