function rbm = train_crbm_updown_LB_v1(dataname, ws, num_bases, pbias,...
    pbias_lb, pbias_lambda, spacing, epsilon, l2reg, batch_size,...
    num_trials, numchannels, sample_size, layer, images_all, model)

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
    
    if exist('model','var')
        % use hidden layer bias from previous training as visible layer bias
        vbias_vec = model.hbias_vec; 
        vbias_vec_fixed = 1;
    elseif ~exist('vbias_vec', 'var') || isempty(vbias_vec)
        vbias_vec = zeros(numchannels,1);
        vbias_vec_fixed = 0;
    end

    if ~exist('hbias_vec', 'var') || isempty(hbias_vec)
        hbias_vec = -0.1*ones(pars.num_bases,1);
    end

    batch_ws = 70; % changed from 100 (2008/07/24)
    imbatch_size = floor(100/batch_size);

    initialmomentum  = 0.5;
    finalmomentum    = 0.9;

    error_history = [];
    sparsity_history = [];

    Winc=0;
    vbiasinc=0;
    hbiasinc=0;
    
    if ~exist('images_all', 'var')
        images_all = sample_images_all(dataname);
    end

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
                imdata_batch = imdata(rowidx, colidx, :);
                imdata_batch = imdata_batch - mean(imdata_batch(:));
                imdata_batch = trim_image_for_spacing_fixconv(imdata_batch, ws, spacing);

%                 if rand()>0.5,
%                     imdata_batch = fliplr(imdata_batch);
%                 end

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
                
                if ~vbias_vec_fixed % only update for the raw image layer
                    vbiasinc = momentum*vbiasinc + epsilon*dv;
                    vbias_vec = vbias_vec + vbiasinc;
                end

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
    %     save(fname_mat, 'W', 'pars', 't', 'vbias_vec', 'hbias_vec', 'error_history', 'sparsity_history');
    %     disp(sprintf('results saved as %s\n', fname_mat));

    %     if mod(t, 10) ==0
    %         fname_timestamp_save = sprintf('%s_%04d.mat', fname_prefix, t);
    %         save(fname_timestamp_save, 'W', 'pars', 't', 'vbias_vec', 'hbias_vec', 'error_history', 'sparsity_history');
    %     end
    end
    
    fname_prefix = sprintf('./results/crbm/crbm_updown_LB_new1h_%s_V1_w%d_b%02d_p%g_pl%g_plambda%g_sp%d_CD_eps%g_l2reg%g_bs%02d_layer%d', dataname, ws, num_bases, pbias, pbias_lb, pbias_lambda, spacing, epsilon, l2reg, batch_size, layer);
    fname_save = sprintf('%s', fname_prefix);
    fname_mat  = sprintf('%s.mat', fname_save);
    fname_out = fname_mat;
    mkdir(fileparts(fname_save));
    
    % save bases
    saveas(gcf, sprintf('%s.png', fname_save));
  
    % output to the next layer
    rbm = struct('W',W,'pars',pars,'vbias_vec',vbias_vec,...
        'hbias_vec',hbias_vec,'error_history',error_history,...
        'sparsity_history',sparsity_history);
    
    % save model
    fname_timestamp_save = sprintf('%s.mat', fname_prefix);
    save(fname_timestamp_save, 'W', 'pars', 't', 'vbias_vec',...
        'hbias_vec', 'error_history', 'sparsity_history',...
        'rbm');
end
    
%% test
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
end

