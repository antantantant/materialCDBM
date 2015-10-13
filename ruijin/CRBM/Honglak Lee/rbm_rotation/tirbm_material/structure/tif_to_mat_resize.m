% Convert .tif to .mat resize
clear
for k = 1:99
    a = imread(sprintf('%d.tif',k));
%     b = imresize(a,36/100);
    b = im2bw(a,0.9);
    c = reshape(b,[1,size(b,1)*size(b,2)]);
    h{k}=c;
end

n = cell2mat(h);
n_BW = reshape(n,[size(b,1)*size(b,2),k])';
% n_test = reshape(n,[k,size(b,1)*size(b,2)]);
save n_BW;
