clear all
close all
clc

data = load('five_cluster.txt');

% parameter record elipson MinPts
% five_cluster 0.4,9
% ThreeCircles 0.3,9
% Twomoons 0.2,5
% spiral 0.5,2

tic
[IDX, isnoise]=DBSCAN(data(:,2:3),0.4,9);
toc
color=linspace(1,10,20)';

scatter(data(:,2),data(:,3),[],color(IDX(:,1)+1));
title(['Number of Clusters ' num2str(max(IDX))]);
% plot(IDX);

% ������