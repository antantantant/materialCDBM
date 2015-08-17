clear
close all
addpath('../structure/');
addpath('../crbm_demo/');

% use the alloy data from Yang Jiao
dataname = 'alloy_scale';

%% First layer
% parameters for first layer RBM
ws = 24; % filter size
num_bases = 24; % number of filters
pbias = 0.002; % sparsity
pbias_lb = 0.002; % sparsity lower bound
pbias_lambda = 5; % ?
spacing = 2; % pooling C?
epsilon = 0.01;
l2reg = 0.01;
batch_size = 10;
num_trial = 100;
numchannels = 1; % input is grayscale
sample_size = 1; % sample size for hidden layer for each visible layer input 
layer = 1;

% train first layer
fname_prefix = sprintf('./results/crbm/crbm_updown_LB_new1h_%s_V1_w%d_b%02d_p%g_pl%g_plambda%g_sp%d_CD_eps%g_l2reg%g_bs%02d_layer%d', dataname, ws, num_bases, pbias, pbias_lb, pbias_lambda, spacing, epsilon, l2reg, batch_size, layer);
fname_timestamp_save = sprintf('%s.mat', fname_prefix);
if exist(fname_timestamp_save, 'file')
    load(fname_timestamp_save);
    hidsample1 = hidsample;
    rbm1 = rbm;
else
    % if haven't trained
    [hidsample1, rbm1] = train_crbm_updown_LB_v1(dataname, ws, num_bases, pbias, pbias_lb,...
        pbias_lambda, spacing, epsilon, l2reg, batch_size,...
        num_trial, numchannels, sample_size, layer);
end

% % show sample hidden layer states
% disp_hidsample = hidsample1{1}+0;
% h=imagesc(sum(disp_hidsample,3),'EraseMode','none',[-1 1]);
% axis image off
% drawnow

%% Second layer
% parameters for second layer RBM
ws = 5; % filter size
numchannels = num_bases; % channel equals #basis from last layer
num_bases = 100; % number of filters
pbias = 0.006; % sparsity
pbias_lb = 0.006; % sparsity lower bound
pbias_lambda = 5; % ?
spacing_in = 2; % backward expansion, should be the same as spacing?
spacing = 2; % pooling C
epsilon = 0.01;
l2reg = 0.01;
l1reg = 1e-4; 
batch_size = 2;
CD_mode = 'exp';
epsdecay = 0.01;
num_trial = 50;
fname_V1_num = 1;

% % train second layer
% [hidsample2, rbm2] = train_crbm_updown_LB_v1(dataname, ws, num_bases, pbias, pbias_lb,...
%     pbias_lambda, spacing, epsilon, l2reg, batch_size,...
%     num_trial, numchannels, sample_size, layer, hidsample1, rbm1);
% train second layer using tirbm
addpath './tirbm/'
addpath './tirbm/tirbm'
train_tirbmnew_updown_BB_v2('pretrained', fname_V1_num, ws, num_bases,...
    pbias, pbias_lb, pbias_lambda, spacing_in, spacing, epsilon, l2reg,...
    l1reg, batch_size, CD_mode, epsdecay, hidsample1, rbm1);

% %% plot second layer filter
% if 0
%     load ./results2/tirbm/tirbmnew_updown_BB_new1j_delta_alloy_V2_w10_b24_p0.006_pl0.006_plambda5_spin2_sp2_exp_eps0.01_epsdecay0.01_l2reg0.01_l1reg0.0001_bs02_20150814T081602.mat
% 
%     % pars.fname_V1
%     % fname_V1 = pars.fname_V1;
%     % V1 = load(fname_V1);
%     V1 = rbm;
%     if length(size(V1.W))<3
%         V1.W = reshape(V1.W, [size(V1.W,1), 1, size(V1.W,2)]);
%     end
% 
%     basisnorm= sqrt(sum(reshape(W, size(W,1)*size(W,2), size(W,3)).^2));
%     [sval sidx] = sort(basisnorm, 'descend');
% 
%     figure(3), display_tirbm_v2_bases_LB_matlab(W(:,:,sidx).*(W(:,:,sidx)>0), V1, pars.spacing_in), colormap gray
%     figure(5), bar(sval)
%     t
% 
%     if 1
%         figure(6), clf
%         numplots = 5;
%         subplot(numplots,1,1), plot(error_history), title('error')
%         subplot(numplots,1,2), plot(hbias_history), title('hbias')
%         subplot(numplots,1,3), plot(sparsity_history), title('sparsity')
%         subplot(numplots,1,4), plot(vbias_history), title('vbias')
%         subplot(numplots,1,5), plot(Wnorm_history), title('Wnorm')
%     end
% end
