%%% test reconstruction based on one layer of CRBM

clear
close all
addpath('../structure/');
addpath('../crbm_demo/');

% use the scaled (0.5*200) alloy data from Dr. Yang Jiao
dataname = 'alloy';

%% First layer
% parameters for first layer RBM
ws = 8; % filter size
num_bases = 36; % number of filters
pbias = 0.002; % sparsity
pbias_lb = 0.002; % sparsity lower bound
pbias_lambda = 5; % ?
spacing = 2; % pooling C
epsilon = 0.01;
l2reg = 0.01;
batch_size = 10;
num_trial = 100;
numchannels = 1; % input is grayscale
sample_size = 1; % sample size for hidden layer for each visible layer input 
layer = 1;

% train first layer
fname_prefix = sprintf('./reconstruction_v1_results/crbm/crbm_updown_LB_new1h_%s_V1_w%d_b%02d_p%g_pl%g_plambda%g_sp%d_CD_eps%g_l2reg%g_bs%02d_layer%d', dataname, ws, num_bases, pbias, pbias_lb, pbias_lambda, spacing, epsilon, l2reg, batch_size, layer);
fname_timestamp_save = sprintf('%s.mat', fname_prefix);
if exist(fname_timestamp_save, 'file')
    load(fname_timestamp_save);
    hidsample1 = hidsample;
    rbm1 = rbm;
else
    % if haven't trained
    rbm1 = train_crbm_updown_LB_v1(dataname, ws, num_bases, pbias, pbias_lb,...
        pbias_lambda, spacing, epsilon, l2reg, batch_size,...
        num_trial, numchannels, sample_size, layer);
end


% sample the hidden layer
% get an image
images_all = sample_images_all(dataname);
image = images_all{1};
image = image - mean(image(:));
image = trim_image_for_spacing_fixconv(image, ws, spacing);

% do convolution/ get poshidprobs
poshidexp = crbm_inference(image, rbm1.W, rbm1.hbias_vec, rbm1.pars);
% poshidstates2 = double(poshidprobs > rand(size(poshidprobs))); 
[poshidstates poshidprobs] = crbm_sample_multrand2(poshidexp, spacing);

% reconstruction of a random input image
negdata = crbm_reconstruct(poshidstates, rbm1.W, rbm1.pars);
display_network(reshape([image(:),negdata(:)],numel(image(:)),1,2));