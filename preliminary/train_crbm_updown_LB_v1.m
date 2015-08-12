function train_crbm_updown_LB_v1(dataname, ws, num_bases, pbias, pbias_lb, pbias_lambda, spacing, epsilon, l2reg, batch_size)

if mod(ws,2)~=0, error('ws must be even number'); end

addpath ../../sparsenet/code
addpath ../../tisc/code/
addpath ../../poggio/code_new

sigma_start = 0.2;
sigma_stop = 0.1;

CD_mode = 'exp';
bias_mode = 'simple';

% Etc parameters
K_CD = 1;

% Initialization
W = [];
vbias_vec = [];
hbias_vec = [];
pars = [];

C_sigm = 1;

% learning
num_trials = 500;

numchannels = 1;

% Initialize variables
if ~exist('pars', 'var') || isempty(pars)
    pars=[];
end

if ~isfield(pars, 'ws'), pars.ws = ws; end
if ~isfield(pars, 'num_bases'), pars.num_bases = num_bases; end
if ~isfield(pars, 'spacing'), pars.spacing = spacing; end

if ~isfield(pars, 'pbias'), pars.pbias = pbias; end
if ~isfield(pars, 'pbias_lb'), pars.pbias_lb = pbias_lb; end
if ~isfield(pars, 'pbias_lambda'), pars.pbias_lambda = pbias_lambda; end
if ~isfield(pars, 'bias_mode'), pars.bias_mode = bias_mode; end

if ~isfield(pars, 'std_gaussian'), pars.std_gaussian = sigma_start; end
if ~isfield(pars, 'sigma_start'), pars.sigma_start = sigma_start; end
if ~isfield(pars, 'sigma_stop'), pars.sigma_stop = sigma_stop; end

if ~isfield(pars, 'K_CD'), pars.K_CD = K_CD; end
if ~isfield(pars, 'CD_mode'), pars.CD_mode = CD_mode; end
if ~isfield(pars, 'C_sigm'), pars.C_sigm = C_sigm; end

if ~isfield(pars, 'num_trials'), pars.num_trials = num_trials; end
if ~isfield(pars, 'epsilon'), pars.epsilon = epsilon; end

disp(pars)


%% Initialize weight matrix, vbias_vec, hbias_vec (unless given)
if ~exist('W', 'var') || isempty(W)
    W = 0.01*randn(pars.ws^2, numchannels, pars.num_bases);
end

if ~exist('vbias_vec', 'var') || isempty(vbias_vec)
    vbias_vec = zeros(numchannels,1);
end

if ~exist('hbias_vec', 'var') || isempty(hbias_vec)
    hbias_vec = -0.1*ones(pars.num_bases,1);
end


batch_ws = 70; % changed from 100 (2008/07/24)
imbatch_size = floor(100/batch_size);

fname_prefix = sprintf('../results/crbm/crbm_updown_LB_new1h_%s_V1_w%d_b%02d_p%g_pl%g_plambda%g_sp%d_CD_eps%g_l2reg%g_bs%02d_%s', dataname, ws, num_bases, pbias, pbias_lb, pbias_lambda, spacing, epsilon, l2reg, batch_size, datestr(now, 30));
fname_save = sprintf('%s', fname_prefix);
fname_mat  = sprintf('%s.mat', fname_save);
fname_out = fname_mat;
mkdir(fileparts(fname_save));
fname_out

initialmomentum  = 0.5;
finalmomentum    = 0.9;

error_history = [];
sparsity_history = [];

Winc=0;
vbiasinc=0;
hbiasinc=0;

images_all = sample_images_all(dataname);

for t=1:pars.num_trials
    % Take a random permutation of the samples
    tic;
    ferr_current_iter = [];
    sparsity_curr_iter = [];

    imidx_batch = randsample(length(images_all), imbatch_size, length(images_all)<imbatch_size);
    for i = 1:length(imidx_batch)
        imidx = imidx_batch(i);
        imdata = images_all{imidx};
        rows = size(imdata,1);
        cols = size(imdata,2);

        for batch=1:batch_size
            % Show progress in epoch
            if 0
                fprintf(1,'epoch %d image %d batch %d\r',t, imidx, batch); 
            end

            rowidx = ceil(rand*(rows-2*ws-batch_ws))+ws + [1:batch_ws];
            colidx = ceil(rand*(cols-2*ws-batch_ws))+ws + [1:batch_ws];
            imdata_batch = imdata(rowidx, colidx);
            imdata_batch = imdata_batch - mean(imdata_batch(:));
            imdata_batch = trim_image_for_spacing_fixconv(imdata_batch, ws, spacing);
            
            if rand()>0.5,
                imdata_batch = fliplr(imdata_batch);
            end
            
            % update rbm
            [ferr dW dh dv poshidprobs poshidstates negdata]= fobj_crbm_CD_LB_sparse(imdata_batch, W, hbias_vec, vbias_vec, pars, CD_mode, bias_mode, spacing, l2reg);
            ferr_current_iter = [ferr_current_iter, ferr];
            sparsity_curr_iter = [sparsity_curr_iter, mean(poshidprobs(:))];

            if t<5,
                momentum = initialmomentum;
            else
                momentum = finalmomentum;
            end

            % update parameters
            Winc = momentum*Winc + epsilon*dW;
            W = W + Winc;

            vbiasinc = momentum*vbiasinc + epsilon*dv;
            vbias_vec = vbias_vec + vbiasinc;

            hbiasinc = momentum*hbiasinc + epsilon*dh;
            hbias_vec = hbias_vec + hbiasinc;
        end
        mean_err = mean(ferr_current_iter);
        mean_sparsity = mean(sparsity_curr_iter);

        if (pars.std_gaussian > pars.sigma_stop) % stop decaying after some point
            pars.std_gaussian = pars.std_gaussian*0.99;
        end

        % figure(1), display_network(W);
        % figure(2), subplot(1,2,1), imagesc(imdata(rowidx, colidx)), colormap gray
        % subplot(1,2,2), imagesc(negdata), colormap gray
    end
    toc;

    error_history(t) = mean(ferr_current_iter);
    sparsity_history(t) = mean(sparsity_curr_iter);

    figure(1), display_network(W);
    % if mod(t,10)==0,
    %    saveas(gcf, sprintf('%s_%04d.png', fname_save, t));
    % end

    fprintf('epoch %d error = %g \tsparsity_hid = %g\n', t, mean(ferr_current_iter), mean(sparsity_curr_iter));
    save(fname_mat, 'W', 'pars', 't', 'vbias_vec', 'hbias_vec', 'error_history', 'sparsity_history');
    disp(sprintf('results saved as %s\n', fname_mat));
  
    if mod(t, 10) ==0
        fname_timestamp_save = sprintf('%s_%04d.mat', fname_prefix, t);
        save(fname_timestamp_save, 'W', 'pars', 't', 'vbias_vec', 'hbias_vec', 'error_history', 'sparsity_history');
    end

end

return

%%

function images = sample_images_all(dataname)

switch lower(dataname),
case 'olshausen',
    fpath = '../data';
    flist = dir(sprintf('%s/*.tif', fpath));
case 'alloy',
    fpath = '../structure';
    flist = dir(sprintf('%s/*.tif', fpath));
end

images = [];
for imidx = 1:min(length(flist), 200)
    fprintf('[%d]', imidx);
    fname = sprintf('%s/%s', fpath, flist(imidx).name);
    im = imread(fname);

    if size(im,3)>1
        im2 = double(rgb2gray(im));
    else
        im2 = double(im);
    end
    ratio = min([512/size(im,1), 512/size(im,2), 1]);
    if ratio<1
        im2 = imresize(im2, [round(ratio*size(im,1)), round(ratio*size(im,2))], 'bicubic');
    end

    im2 = crbm_whiten_olshausen2(im2);
    % im2 = crbm_whiten_olshausen1_contrastnorm(im2, 32, true, 0.6);
    % im2 = preprocess_image(im2, 'whiten'); % whiten the image before resizing
    im2 = im2-mean(mean(im2));
    im2 = im2/sqrt(mean(mean(im2.^2)));
    imdata = im2;
    imdata = sqrt(0.1)*imdata; % just for some trick??
    images{length(images)+1} = imdata;

    if 0
        imagesc(imdata), axis off equal, colormap gray
    end
end
fprintf('\n');


return


function [poshidexp2] = crbm_inference(imdata, W, hbias_vec, pars)

ws = sqrt(size(W,1));
numbases = size(W,3);
numchannel = size(W,2);

poshidprobs2 = zeros(size(imdata,1)-ws+1, size(imdata,2)-ws+1, numbases);
poshidexp2 = zeros(size(imdata,1)-ws+1, size(imdata,2)-ws+1, numbases);
for c=1:numchannel
    H = reshape(W(end:-1:1, c, :),[ws,ws,numbases]);
    poshidexp2 = poshidexp2 + conv2_mult(imdata(:,:,c), H, 'valid');
end

for b=1:numbases
    poshidexp2(:,:,b) = 1/(pars.std_gaussian^2).*(poshidexp2(:,:,b) + hbias_vec(b));
    poshidprobs2(:,:,b) = 1./(1 + exp(-poshidexp2(:,:,b)));
end

return


function vishidprod2 = crbm_vishidprod_fixconv(imdata, H, ws)

numchannels = size(imdata,3);
numbases = size(H,3);

% tic
% TODO: single channel version is not implemented yet.. Might need to
% modify mexglx file
selidx1 = size(H,1):-1:1;
selidx2 = size(H,2):-1:1;
vishidprod2 = zeros(ws,ws,numchannels,numbases);

if numchannels==1
    vishidprod2 = conv2_mult(imdata, H(selidx1, selidx2, :), 'valid');
else
    for b=1:numbases
        vishidprod2(:,:,:,b) = conv2_mult(imdata, H(selidx1, selidx2, b), 'valid');
    end
end

vishidprod2 = reshape(vishidprod2, [ws^2, numchannels, numbases]);

return



function negdata = crbm_reconstruct(S, W, pars)

ws = sqrt(size(W,1));
patch_M = size(S,1);
patch_N = size(S,2);
numchannels = size(W,2);
numbases = size(W,3);

S2 = S;
negdata2 = zeros(patch_M+ws-1, patch_N+ws-1, numchannels);

for b = 1:numbases,
    H = reshape(W(:,:,b),[ws,ws,numchannels]);
    negdata2 = negdata2 + conv2_mult(S2(:,:,b), H, 'full');
end

negdata = 1*negdata2;

return


function [ferr dW_total dh_total dv_total poshidprobs poshidstates negdata] = ...
    fobj_crbm_CD_LB_sparse(imdata, W, hbias_vec, vbias_vec, pars, CD_mode, bias_mode, spacing, l2reg)

ws = sqrt(size(W,1));

%%%%%%%%% START POSITIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% do convolution/ get poshidprobs
poshidexp = crbm_inference(imdata, W, hbias_vec, pars);
% poshidstates2 = double(poshidprobs > rand(size(poshidprobs))); 
[poshidstates poshidprobs] = crbm_sample_multrand2(poshidexp, spacing);

posprods = crbm_vishidprod_fixconv(imdata, poshidprobs, ws);
poshidact = squeeze(sum(sum(poshidprobs,1),2));

%%%%%%%%% START NEGATIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
neghidstates = poshidstates;
for j=1:pars.K_CD  %% pars.K_CD-step contrastive divergence
    negdata = crbm_reconstruct(neghidstates, W, pars);
    % neghidprobs = crbm_inference(negdata, W, hbias_vec, pars);
    % neghidstates = neghidprobs > rand(size(neghidprobs));
    neghidexp = crbm_inference(negdata, W, hbias_vec, pars);
    [neghidstates neghidprobs] = crbm_sample_multrand2(neghidexp, spacing);
    
end
negprods = crbm_vishidprod_fixconv(negdata, neghidprobs, ws);
neghidact = squeeze(sum(sum(neghidprobs,1),2));

ferr = mean( (imdata(:)-negdata(:)).^2 );

if 0
    figure(1), display_images(imdata)
    figure(2), display_images(negdata)

    figure(3), display_images(W)
    figure(4), display_images(posprods)
    figure(5), display_images(negprods)

    figure(6), display_images(poshidstates)
    figure(7), display_images(neghidstates)
end


%%%%%%%%% UPDATE WEIGHTS AND BIASES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
if strcmp(bias_mode, 'none')
    dhbias = 0;
    dvbias = 0;
    dW = 0;
elseif strcmp(bias_mode, 'simple')
    dhbias = squeeze(mean(mean(poshidprobs,1),2)) - pars.pbias;
    dvbias = 0;
    dW = 0;
elseif strcmp(bias_mode, 'hgrad')
    error('hgrad not yet implemented!');
elseif strcmp(bias_mode, 'Whgrad')
    error('Whgrad not yet implemented!');
else
    error('wrong adjust_bias mode!');
end

numcases1 = size(poshidprobs,1)*size(poshidprobs,2);
% dW_total = (posprods-negprods)/numcases - l2reg*W - weightcost_l1*sign(W) - pars.pbias_lambda*dW;
dW_total1 = (posprods-negprods)/numcases1;
dW_total2 = - l2reg*W;
dW_total3 = - pars.pbias_lambda*dW;
dW_total = dW_total1 + dW_total2 + dW_total3;

dh_total = (poshidact-neghidact)/numcases1 - pars.pbias_lambda*dhbias;

dv_total = 0; %dv_total';

if 0
    fprintf('||W||=%g, ||dWprod|| = %g, ||dWl2|| = %g, ||dWsparse|| = %g\n', sqrt(sum(W(:).^2)), sqrt(sum(dW_total1(:).^2)), sqrt(sum(dW_total2(:).^2)), sqrt(sum(dW_total3(:).^2)));
end

return



function [H HP] = crbm_sample_multrand2(poshidexp, spacing)
% poshidexp is 3d array
poshidprobs = exp(poshidexp);
poshidprobs_mult = zeros(spacing^2+1, size(poshidprobs,1)*size(poshidprobs,2)*size(poshidprobs,3)/spacing^2);
poshidprobs_mult(end,:) = 1;
% TODO: replace this with more realistic activation, bases..
for c=1:spacing
    for r=1:spacing
        temp = poshidprobs(r:spacing:end, c:spacing:end, :);
        poshidprobs_mult((c-1)*spacing+r,:) = temp(:);
    end
end

% [S P] = multrand2(poshidprobs_mult');
[S1 P1] = multrand2(poshidprobs_mult');
S = S1';
P = P1';
clear S1 P1

% convert back to original sized matrix
H = zeros(size(poshidexp));
HP = zeros(size(poshidexp));
for c=1:spacing
    for r=1:spacing
        H(r:spacing:end, c:spacing:end, :) = reshape(S((c-1)*spacing+r,:), [size(H,1)/spacing, size(H,2)/spacing, size(H,3)]);
        HP(r:spacing:end, c:spacing:end, :) = reshape(P((c-1)*spacing+r,:), [size(H,1)/spacing, size(H,2)/spacing, size(H,3)]);
    end
end


return


function im2 = trim_image_for_spacing_fixconv(im2, ws, spacing)
% % Trim image so that it matches the spacing.
if mod(size(im2,1)-ws+1, spacing)~=0
    n = mod(size(im2,1)-ws+1, spacing);
    im2(1:floor(n/2), : ,:) = [];
    im2(end-ceil(n/2)+1:end, : ,:) = [];
end
if mod(size(im2,2)-ws+1, spacing)~=0
    n = mod(size(im2,2)-ws+1, spacing);
    im2(:, 1:floor(n/2), :) = [];
    im2(:, end-ceil(n/2)+1:end, :) = [];
end
return


function y = conv2_mult(a, B, convopt)
y = [];
for i=1:size(B,3)
    y(:,:,i) = conv2(a, B(:,:,i), convopt);
end
return


function test
if 0
%%
matlab -nodisplay
cd /afs/cs/u/hllee/visionnew/trunk/belief_nets/code_nips08
addpath crbm

figure(1), display_network(W)
% figure(2), hist(W(:), 30)

pars.std_gaussian

t

%%
end
return

