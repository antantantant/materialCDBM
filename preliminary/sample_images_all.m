%% sample_images_all
function images = sample_images_all(dataname)

    switch lower(dataname),
    case 'olshausen',
        fpath = '../data';
        flist = dir(sprintf('%s/*.tif', fpath));
    case 'alloy',
        fpath = '../structure';
        flist = dir(sprintf('%s/*.tif', fpath));
    case 'alloy_scale',
        fpath = '../structure';
        flist = dir(sprintf('%s/*_scale05.tif', fpath));
    case 'kyoto',
        fpath = './kyoto/gray8bit';
        flist = dir(sprintf('%s/*.png', fpath));        
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
end