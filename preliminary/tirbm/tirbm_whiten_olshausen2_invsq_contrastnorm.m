function im_out = tirbm_whiten_olshausen1_contrastnorm1(im, Qnn, D, noenhance)

global global_Qss_freq global_Qnn global_filt_orig; % Use global variable to speed up the the code
opt_global = true;

if ~exist('D', 'var'), D = 16; end

if size(im,3)>1, im = rgb2gray(im); end
im = double(im);

im = im - mean(im(:));
im = im./std(im(:));

N1 = size(im, 1);
N2 = size(im, 2);

% [fx fy]=meshgrid(-N1/2:N1/2-1, -N2/2:N2/2-1);
% rho=sqrt(fx.*fx+fy.*fy)';

% f_0=0.4*mean([N1,N2]);
% filt=rho.*exp(-(rho/f_0).^4);
% filt=rho./(1+5*(rho/f_0).^2);

if ~opt_global
    % load Qss_kyoto.mat Qss Qss_freq
    load Qss_kyoto.mat Qss_freq

    filt_new = (sqrt(Qss_freq)./(Qss_freq+Qnn));
else
    if isempty(global_Qss_freq)
        load Qss_kyoto.mat Qss_freq
        global_Qss_freq = Qss_freq;
    end
    Qss_freq = global_Qss_freq;
    
    if isempty(global_Qnn) || Qnn ~= global_Qnn
        global_Qnn = Qnn;
        global_filt_orig = (sqrt(Qss_freq)./(Qss_freq+Qnn));
    end
    filt_new = global_filt_orig;
end

% Qss_freq = abs(fftshift(fft2(Qss)));

% load Qss_kyoto.mat Qss_freq
% global_filt_orig = (sqrt(Qss_freq)./(Qss_freq+Qnn));
filt_new = imresize(filt_new, [N1, N2], 'bicubic');

If=fft2(im);
imw=real(ifft2(If.*fftshift(filt_new)));

% contrast normalization
[x y] = meshgrid(-D/2:D/2);
G = exp(-0.5*((x.^2+y.^2)/(D/2)^2));
G = G/sum(G(:));
imv = conv2(imw.^2,G,'same');
imv2 = conv2(ones(size(imw)),G,'same');
% imn = imw./sqrt(imv);
if ~noenhance
    imn = imw; % This is modified version (no contrast normalization)
%     imn = imw./sqrt(imv);
%     imn = imw./sqrt(imv).*sqrt(imv2);
else
    cutoff = quantile(sqrt(imv(:)), 0.3); 
    imn = imw./max(sqrt(imv), cutoff);
%     cutoff = quantile(sqrt(imv(:)./imv2(:)), 0.3); 
%     imn = imw./max(sqrt(imv)./sqrt(imv2), cutoff);
end

im_out = imn/std(imn(:)); % 0.1 is the same factor as in make-your-own-images

return

