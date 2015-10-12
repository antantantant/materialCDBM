clear
close all

dataname = 'Olshausen';
ws = 10;
num_bases = 24;
pbias = 0.002;
pbias_lb = 0.002;
pbias_lambda = 5;
spacing = 2;
epsilon = 0.01;
l2reg = 0.01;
batch_size = 2;

train_crbm_updown_LB_v1(dataname, ws, num_bases, pbias, pbias_lb, pbias_lambda, spacing, epsilon, l2reg, batch_size);


