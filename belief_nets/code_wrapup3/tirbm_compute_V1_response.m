function [H HP Hc HPc imdata_v0] = tirbm_compute_V1_response(im, V1, spacing_in, imsize, D, ws_pad, noiselevel)

if ~exist('noiselevel', 'var')
    noiselevel = 0.5;
end

%%
% im = zeros(size(im2)); % this is a dummy variable...
if size(im,3)>1, im2 = double(rgb2gray(im));
else im2 = double(im);
end

if ws_pad
    padval = (mean(mean(im2(:, [1,size(im2,2)]))) + mean(mean(im2([1,size(im2,1)],:))))/2;
    im2 = padarray(im2, [ws_pad, ws_pad], padval);
end

ratio = min([imsize/size(im,1), imsize/size(im,2), 1]);
im2 = imresize(im2, [round(ratio*size(im,1)), round(ratio*size(im,2))], 'bicubic');

% im2_new = tirbm_whiten_olshausen1_contrastnorm(im2, D, true, 0.5);
% im2_new = tirbm_whiten_olshausen2_contrastnorm(im2, D, true);
% im2_new = tirbm_whiten_olshausen2_invsq_contrastnorm(im2, 0.4, D, true);
im2_new = tirbm_whiten_olshausen2_invsq_contrastnorm(im2, noiselevel, D, true);
% im2_new = tirbm_whiten_olshausen2_invsq_contrastnorm(im2, 1, D, true);
% im2_new = tirbm_whiten_olshausen2_invsq(im2, 1);

im2 = im2_new;
im2 = im2-mean(mean(im2));
im2 = im2/sqrt(mean(mean(im2.^2)));

ws = sqrt(size(V1.W, 1));
im2 = trim_image_for_spacing_fixconv(im2, ws, spacing_in);

imdata_v0 = im2/1.5;

% [poshidprobs2_old poshidexp2_old] = tirbm_inference_updown_fixconv(imdata_v0, V1.W, V1.hbias_vec, V1.pars);
% poshidexp2 = tirbm_inference_updown_fixconv_fast(imdata_v0, V1.W, V1.hbias_vec, V1.pars);
poshidexp2 = tirbm_inference(imdata_v0, V1.W, V1.hbias_vec, V1.pars);

% [H_old HP_old Hc_old HPc_old] = tirbm_sample_multrand2(poshidexp2, spacing_in);
[H HP Hc HPc] = tirbm_sample_multrand2_fast(poshidexp2, spacing_in);

%%

if 0
%%
    negdata2 = tirbm_reconstruct_LB(HP, V1.W, V1.pars);
    negdata2_expand = tirbm_reconstruct_LB_expand2(HPc, V1.W, spacing_in, V1.pars)/spacing_in;

%     figure(1), clf;
    figure
    subplot(2,2,1), imagesc(im), colormap gray; title('image')
    subplot(2,2,2), imagesc(imdata_v0), colormap gray; title('v0')
    subplot(2,2,3), imagesc(negdata2), colormap gray; title('v1')
    subplot(2,2,4), imagesc(negdata2_expand), colormap gray; title('v1c')
    whos imresp
%%
end

return

function negdata_expand = tirbm_reconstruct_LB_expand2(poshid_reduced, W, expandfactor, pars)

poshid_expand = imresize(poshid_reduced, expandfactor, 'bicubic');
negdata_expand = tirbm_reconstruct_LB(poshid_expand, W, pars);

return
