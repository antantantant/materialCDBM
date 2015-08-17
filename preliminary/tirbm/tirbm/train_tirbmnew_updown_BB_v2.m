% only use this to train the second layer
% use 'pretrained' when images_all and model exist
function train_tirbmnew_updown_BB_v2(dataname, fname_V1_num, ws, num_bases,...
    pbias, pbias_lb, pbias_lambda, spacing_in, spacing, epsilon, l2reg,...
    l1reg, batch_size, CD_mode, epsdecay, images_all, model)

if mod(ws,2)~=0, error('ws must be even number'); end

% spacing = 1;

opt_caltech = false;
if isnumeric(dataname)
	opt_caltech = true;
	classes = dataname;
end

sigma_start = 0.2;
sigma_stop = 0.2;

% CD_mode = 'exp';
% CD_mode = 'mf'; % mean field
bias_mode = 'simple';
% L2_lambda = 5;

% Etc parameters
K_CD = 1;
C_sigm = 1;

% Initialization
W = [];
vbias_vec = [];
hbias_vec = [];
pars = [];

% learning
num_trials = 50;
% epsilon = 0.002; %0.002;

% Fixed parameters
K_SAVEON = 1; % save results on every K_SAVEON epochs

% pbias_lb = 0.001;
% batch_size = 5;

if exist('model','var')
    V1 = model;
else
    fname_V1 = get_fname_V1(fname_V1_num);
    V1 = load(fname_V1);
    pars.fname_V1 = fname_V1;
end


if length(size(V1.W))<3, V1.W = reshape(V1.W, [size(V1.W,1), 1, size(V1.W,2)]); end
numchannels = size(V1.W,3);

% Initialize variables
if ~exist('pars', 'var') || isempty(pars)
    pars=[];
end

if ~isfield(pars, 'ws'), pars.ws = ws; end
if ~isfield(pars, 'num_bases'), pars.num_bases = num_bases; end
if ~isfield(pars, 'spacing'), pars.spacing = spacing; end
if ~isfield(pars, 'spacing_in'), pars.spacing_in = spacing_in; end

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
if ~isfield(pars, 'l2reg'), pars.l2reg = l2reg; end
if ~isfield(pars, 'l1reg'), pars.l1reg = l1reg; end

if ~isfield(pars, 'batch_size'), pars.batch_size = batch_size; end

% TODO
if opt_caltech
	pars.classes = classes;
end

disp(pars)

%% Initialize weight matrix, vbias_vec, hbias_vec (unless given)
if ~exist('W', 'var') || isempty(W)
    W = 0.02*randn(pars.ws^2, numchannels, pars.num_bases);
end

if ~exist('vbias_vec', 'var') || isempty(vbias_vec)
    vbias_vec = -0.1*ones(numchannels,1);
    % vbias_vec = -0*ones(numchannels,1); % TEST version
end

if ~exist('hbias_vec', 'var') || isempty(hbias_vec)
    hbias_vec = -0.1*ones(pars.num_bases,1);
end

% batch_size = 4;
batch_ws = 50; % changed from 70 -> 50 
imbatch_size = floor(100/batch_size);

if ~opt_caltech
	fname_prefix = sprintf('./results2/tirbm/tirbmnew_updown_BB_new1j_delta_%s_V2_w%d_b%02d_p%g_pl%g_plambda%g_spin%d_sp%d_%s_eps%g_epsdecay%g_l2reg%g_l1reg%g_bs%02d_%s', dataname, ws, num_bases, pbias, pbias_lb, pbias_lambda, spacing_in, spacing, CD_mode, epsilon, epsdecay, l2reg, l1reg, batch_size, datestr(now, 30)); % TEST version
else
	fname_prefix = sprintf('./results2/tirbm/tirbmnew_updown_BB_new1j_delta_caltech_c%d-%d[%d]_V2_w%d_b%02d_p%g_pl%g_plambda%g_spin%d_sp%d_%s_eps%g_epsdecay%g_l2reg%g_l1reg%g_bs%02d_%s', min(classes), max(classes), length(classes), ws, num_bases, pbias, pbias_lb, pbias_lambda, spacing_in, spacing, CD_mode, epsilon, epsdecay, l2reg, l1reg, batch_size, datestr(now, 30)); % TEST version
end

fname_save = sprintf('%s', fname_prefix);
fname_mat  = sprintf('%s.mat', fname_save);
fname_out = fname_mat;
mkdir(fileparts(fname_save));
fname_out

initialmomentum  = 0.5;
finalmomentum    = 0.9;

error_history = [];
sparsity_history = [];
hbias_history = [];
vbias_history = [];
Wnorm_history = [];

Winc=0;
vbiasinc=0;
hbiasinc=0;

if ~strcmp(dataname,'pretrained')
    if ~opt_caltech
        images_all = sample_images_all_v1([], dataname, V1, spacing_in);
    end

    % TEMPORARY: just picked them out
    if opt_caltech,
        if length(classes)>4
        else
            images_all = sample_images_all_v1_caltech(classes, V1, spacing_in, 100);
        end
    end
end

for t=1:pars.num_trials
    epsilon = pars.epsilon/(1+epsdecay*t);
    
    if ~strcmp(dataname,'pretrained')
        % TEMPORARY: just picked them out
        if opt_caltech,
            if length(classes)>4
                images_all = sample_images_all_v1_caltech(classes(randsample(length(classes),1)), V1, spacing_in, 20);
            else
            end
        elseif mod(t,10)==0
            images_all = sample_images_all_v1([], dataname, V1, spacing_in);
        end
    end
        
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
            fprintf(1,'epoch %d image %d batch %d\r',t, imidx, batch); 

            rowstart = ceil(rand*(max(rows-2*ws-batch_ws,0)))+ws;
            rowidx = rowstart:min(rowstart+batch_ws-1, rows-ws);
            colstart = ceil(rand*(max(cols-2*ws-batch_ws, 0)))+ws;
            colidx = colstart:min(colstart+batch_ws-1, cols-ws);
            if length(rowidx)<2*ws || length(colidx)<2*ws
                rowidx = 1:rows;
                colidx = 1:cols;
            end

			imresp = imdata(rowidx, colidx, :)+0; % make sure to convert logical to double
            imresp = trim_image_for_spacing_fixconv(imresp, ws, spacing);

            % update rbm
            [ferr dW dh dv poshidprobs poshidstates negdata]= fobj_tirbm_CD_BB_sparse(imresp, W, hbias_vec, vbias_vec, pars, CD_mode, bias_mode, spacing, l2reg, l1reg);
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
        
        if mod(i, 10)==0, fprintf('err= %g, sparsity= %g mean(hbias)= %g,  mean(vbias)= %g\n', mean_err, mean_sparsity, mean(hbias_vec), mean(vbias_vec)); end
    end
    toc;

    if (pars.std_gaussian > pars.sigma_stop) % stop decaying after some point
        pars.std_gaussian = pars.std_gaussian*0.99;
    end
    stdev = pars.std_gaussian;
    
    error_history(t) = mean(ferr_current_iter);
    sparsity_history(t) = mean(sparsity_curr_iter);
    hbias_history(t) = mean(hbias_vec);
    vbias_history(t) = mean(vbias_vec);
    Wnorm_history(t) = sqrt(sum(W(:).^2)/num_bases);
    
    fprintf('epoch %d error = %g \tsparsity_hid = %g\n', t, mean(ferr_current_iter), mean(sparsity_curr_iter));
    if mod(t, K_SAVEON)==0
        save(fname_mat, 'W', 'pars', 't', 'vbias_vec', 'hbias_vec', 'error_history', 'sparsity_history', 'hbias_history', 'vbias_history', 'Wnorm_history');
        disp(sprintf('results saved as %s\n', fname_mat));
    end
  
    if mod(t, 20) ==0
        fname_timestamp_save = sprintf('%s_%04d.mat', fname_prefix, t);
        save(fname_timestamp_save, 'W', 'pars', 't', 'vbias_vec', 'hbias_vec', 'error_history', 'sparsity_history');
    end

end


%% plot 2nd layer filter

V1 = model;
if length(size(V1.W))<3
    V1.W = reshape(V1.W, [size(V1.W,1), 1, size(V1.W,2)]);
end

basisnorm= sqrt(sum(reshape(W, size(W,1)*size(W,2), size(W,3)).^2));
[sval sidx] = sort(basisnorm, 'descend');

figure(3), display_tirbm_v2_bases_LB_matlab(W(:,:,sidx).*(W(:,:,sidx)>0), V1, pars.spacing_in), colormap gray
figure(5), bar(sval)
t

if 1
    figure(6), clf
    numplots = 5;
    subplot(numplots,1,1), plot(error_history), title('error')
    subplot(numplots,1,2), plot(hbias_history), title('hbias')
    subplot(numplots,1,3), plot(sparsity_history), title('sparsity')
    subplot(numplots,1,4), plot(vbias_history), title('vbias')
    subplot(numplots,1,5), plot(Wnorm_history), title('Wnorm')
end


return



function fname_V1 = get_fname_V1(fname_V1_num)
switch fname_V1_num,
    case 1,
        fname_V1 = 'tirbm_updown_LB_new1h_rot_Olshausen_V1_w10_b24.mat';
end

return


function images = sample_images_all_v1_caltech(c, V1, spacing_in, num_images)

base= '~/robo/brain/data/101_ObjectCategories/';
baseDir= dir(base);

images = [];
for i=1:length(c)
    images = sample_images_all_v1(images, baseDir(c(i)+2).name, V1, spacing_in, num_images);
end
return

function images = sample_images_all_v1(images, dataname, V1, spacing_in, numimages)

if ~exist('numimages', 'var'), numimages = 200; end

[fpath, flist, imsize, D, ws_pad] = tirbm_get_image_info(dataname, spacing_in);

% images = [];
for imidx = randsample(1:length(flist), min(length(flist), numimages))
%%
    fprintf('[%d]', imidx);
    fname = sprintf('%s/%s', fpath, flist(imidx).name);
    im = imread(fname);

    [H HP Hc HPc] = tirbm_compute_V1_response(im, V1, spacing_in, imsize, D, ws_pad);

    BUF = 2;
    imresp = HPc;
    imresp(1:BUF,:,:)=0;
    imresp(:, 1:BUF,:)=0;
    imresp(end-BUF+1:end,:,:)=0;
    imresp(:, end-BUF+1:end,:)=0;

    images{length(images)+1} = imresp;
end
fprintf('\n');

return



function negdata = tirbm_reconstruct_BB(S, W, vbias_vec, pars)

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

negdata = zeros(patch_M+ws-1, patch_N+ws-1, numchannels);
for c = 1:numchannels
    negdata(:,:,c) = 1./(1 + exp(-pars.C_sigm/(pars.std_gaussian^2).*(negdata2(:,:,c) + vbias_vec(c))));
end

return



function [ferr dW_total dh_total dv_total poshidprobs poshidstates negdata] = ...
    fobj_tirbm_CD_BB_sparse(imdata, W, hbias_vec, vbias_vec, pars, CD_mode, bias_mode, spacing, l2reg, l1reg)

ws = sqrt(size(W,1));

%%%%%%%%% START POSITIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% do convolution/ get poshidprobs
poshidexp = tirbm_inference(imdata, W, hbias_vec, pars);
[poshidstates poshidprobs] = tirbm_sample_multrand2_fast(poshidexp, spacing);

% posprods_old = tirbm_vishidprod_fixconv(imdata, poshidprobs, ws);
posprods = tirbm_vishidprod_fixconv(imdata, poshidprobs, ws);

poshidact = squeeze(sum(sum(poshidprobs,1),2));
posvisact   = squeeze(sum(sum(imdata,1),2));

%%%%%%%%% START NEGATIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
neghidstates = poshidstates;
for j=1:pars.K_CD  %% pars.K_CD-step contrastive divergence
    negdata = tirbm_reconstruct_BB(neghidstates, W, vbias_vec, pars);
    neghidexp = tirbm_inference(negdata, W, hbias_vec, pars);
    [neghidstates neghidprobs] = tirbm_sample_multrand2_fast(neghidexp, spacing);
end
negprods = tirbm_vishidprod_fixconv(negdata, neghidprobs, ws);

neghidact = squeeze(sum(sum(neghidprobs,1),2));
negvisact = squeeze(sum(sum(negdata,1),2)); 

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
    avg_hidprobs = squeeze(mean(mean(poshidprobs,1),2));
    dhbias = 0;
    dhbias = dhbias + (avg_hidprobs>pars.pbias).*(avg_hidprobs-pars.pbias);
    dhbias = dhbias + (avg_hidprobs<pars.pbias_lb).*(avg_hidprobs-pars.pbias_lb); % This is a bit of hack..
    % dhbias = squeeze(mean(mean(poshidprobs,1),2)) - pars.pbias;
    % dhbias(dhbias<0) = 0; % only penalize for large activation..
    dvbias = 0;
    dW = 0;
elseif strcmp(bias_mode, 'hgrad')
    error('hgrad not yet implemented!');
elseif strcmp(bias_mode, 'Whgrad')
    error('Whgrad not yet implemented!');
else
    error('wrong adjust_bias mode!');
end

numcases1 = (size(poshidprobs,1))*(size(poshidprobs,2));
numcases2 = size(imdata,1)*size(imdata,2);
% dW_total = (posprods-negprods)/numcases - l2reg*W - weightcost_l1*sign(W) - pars.L2_lambda*dW;
dW_total1 = (posprods-negprods)/numcases1;
dW_total2 = - l2reg*W;
dW_total4 = - l1reg*sign(W);
dW_total3 = - pars.pbias_lambda*dW;
% dv_total = (posvisact-negvisact)/numcases - pars.L2_lambda*dvbias;
dW_total = dW_total1 + dW_total2 + dW_total3 + dW_total4;

dh_total = (poshidact-neghidact)/numcases1 - pars.pbias_lambda*dhbias;
dv_total = (posvisact-negvisact)/numcases2;

fprintf('||W||=%g, ||dWprod|| = %g, ||dWl2|| = %g, ||dWsparse|| = %g\n', sqrt(sum(W(:).^2)), sqrt(sum(dW_total1(:).^2)), sqrt(sum(dW_total2(:).^2)), sqrt(sum(dW_total3(:).^2)));

%dv_total = 0; %dv_total';
%dh_total = dh_total;

return




function test
if 1
%%
% The script below will visualize the second layer bases

% load fname_timestamp_save

% pars.fname_V1
% fname_V1 = pars.fname_V1;
% V1 = load(fname_V1);
V1 = model;
if length(size(V1.W))<3
    V1.W = reshape(V1.W, [size(V1.W,1), 1, size(V1.W,2)]);
end

basisnorm= sqrt(sum(reshape(W, size(W,1)*size(W,2), size(W,3)).^2));
[sval sidx] = sort(basisnorm, 'descend');

figure(3), display_tirbm_v2_bases_LB_matlab(W(:,:,sidx).*(W(:,:,sidx)>0), V1, pars.spacing_in), colormap gray
figure(5), bar(sval)
t

if 1
    figure(6), clf
    numplots = 5;
    subplot(numplots,1,1), plot(error_history), title('error')
    subplot(numplots,1,2), plot(hbias_history), title('hbias')
    subplot(numplots,1,3), plot(sparsity_history), title('sparsity')
    subplot(numplots,1,4), plot(vbias_history), title('vbias')
    subplot(numplots,1,5), plot(Wnorm_history), title('Wnorm')
end

%%
end

return

