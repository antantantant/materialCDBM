% Convert .tif to .mat resize
clear
for k = 1:99
    a = imread(sprintf('%d.tif',k));
%     b = imresize(a,36/100);
    b = im2bw(a,0.9); % for binary
    b = reshape(a,[1,size(a,1)*size(a,2)]);
    h{k}=b;
end

n = cell2mat(h);
n_test3 = reshape(n,[size(b,1)*size(b,2),k])';
% n_test3 = reshape(n,[k,size(b,1)*size(b,2)]);
save n_test3;
