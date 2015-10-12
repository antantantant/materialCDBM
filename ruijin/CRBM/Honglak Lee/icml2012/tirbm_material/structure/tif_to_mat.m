% Convert .tif to .mat
for k = 1:99
    a=imread(sprintf('%d_scale05.tif',k));
    b=reshape(a,[1,size(a,1)*size(a,2)]);
    h{k}=b;    
end
m = cell2mat(h);
m = reshape(m,[k,size(a,1)*size(a,2)]);
save m;
