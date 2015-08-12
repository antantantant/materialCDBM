addpath('../structure/');
addpath('../toolbox_graph/');

% read .vtk file from /structure
for id = 0:99
    fname = [num2str(id),'.vtk'];
    fid = fopen(fname);

    tline = fgets(fid);
    count = 1;
    NV = 200; % raw image dimension
    mat = zeros(NV^2,1);
    while ischar(tline)
        count = count + 1;
        if count > 11
            mat(count-11) = str2num(tline);
        end
        tline = fgets(fid);
    end
    fclose(fid);
    mat = reshape(mat,200,200);
    mat = mat/2; % convert to grayscale
    output = ['../structure/',num2str(id),'.tif'];
    imwrite(mat,output);
end
