function display_tirbm_v2_bases_LB_matlab(W, V1, expandfactor, opt_nonneg, cols)

if ~exist('opt_nonneg', 'var'), opt_nonneg = false; end
if ~exist('expandfactor', 'var'), expandfactor = 4; end

images = [];
% expandfactor = 4;
ws = sqrt(size(W,1));
for i=1:size(W, 3)
    poshid_reduced = reshape(W(:,:,i), [ws, ws, size(W,2)]);
    poshid_expand = imresize(poshid_reduced, expandfactor, 'bicubic');
    
    % negdata_expand = tirbm_reconstruct_LB_fixconv(poshid_expand, V1.W, V1.pars);
    negdata_expand = tirbm_reconstruct_LB(poshid_expand, V1.W, V1.pars);
    % negdata_expand = tirbm_reconstruct(poshid_expand, V1.W, V1.pars, opt_nonneg);
    if isempty(images), images = zeros(size(negdata_expand,1), size(negdata_expand,2), size(W,3)); end
    images(:,:,i) = negdata_expand;
end

% A = zeros(size(images,1)*size(images,2), size(A,3));
if exist('cols', 'var')
    % display_images(images, false, cols);
    display_network(reshape(images, size(images,1)*size(images,2), size(images,3)), true, true, cols, true);
else
    % display_images(images);
    display_network(reshape(images, size(images,1)*size(images,2), size(images,3)), true, true);
end

return



function negdata = tirbm_reconstruct_LB(S, W, pars)

ws = sqrt(size(W,1));
patch_M = size(S,1);
patch_N = size(S,2);
numchannels = size(W,2);
numbases = size(W,3);

% Note: Reconstruction was off by a few pixels in the original code (above
% versions).. I fixed this as below:
S2 = zeros(size(S));
S2(2:end,2:end,:) = S(1:end-1,1:end-1,:);
negdata2 = zeros(patch_M, patch_N, numchannels);
if numchannels == 1
    H = reshape(W,[ws,ws,numbases]);
    negdata2 = sum(conv2_mult_pairwise(S2, H, 'same'),3);
else
    for b = 1:numbases,
        H = reshape(W(:,:,b),[ws,ws,numchannels]);
        negdata2 = negdata2 + conv2_mult(S2(:,:,b), H, 'same');
    end
end

negdata = pars.C_sigm*negdata2;
% imagesc(negdata); colormap gray

return
