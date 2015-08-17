function [fpath, flist, imsize, D, ws_pad] = tirbm_get_image_info(dataname, spacing_in)

ws_pad = 0;
switch lower(dataname),
case 'kyoto',
    fpath = '../data/kyoto/gray8bit';
    flist = dir(sprintf('%s/*.png', fpath));
    imsize = 512;
    D = 32;
otherwise, 
    fpath = sprintf('../data/101_ObjectCategories/%s/', dataname);
    if exist(fpath, 'dir')
        flist = dir(sprintf('%s/*.jpg', fpath));
        imsize = 160; 
        if spacing_in==3, imsize = 180; end
        D = 20;
    else
        error('dataname is not yet supported\n');
    end
end

return
