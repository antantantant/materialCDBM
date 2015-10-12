% Convert .tif to .mat resize
for k = 1:99
    a = imread(sprintf('%d_scale05.tif',k));
    b = imresize(a,36/100);
    c = reshape(b,[1,size(b,1)*size(b,2)]);
    h{k}=c;
end

n = cell2mat(h);
n = reshape(n,[k,size(b,1)*size(b,2)]);
save n;
