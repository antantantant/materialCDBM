clear;clc
sample = imread('test_2.tif');
[rows, cols, channels] = size(sample);
% sample = imresize(sample, 1/16);
sample_blured = imread('test_2_scale05.tif');
blured_resize = imresize(sample_blured, 2);

for n = 1:200
for ii = 1:179
    for jj = 1:179
       patch = blured_resize(ii:ii+20,jj:jj+20);
%        miu_x = sum(sum(patch))/length(patch);
%        sigma_x = 
       
    for i=1:180
        for j=1:180
% calculate the similarity between two patched

        patch_sample = sample(i:i+20,j:j+20);
        A(i,j)=sum(sum(abs(patch - patch_sample)));
        
        %%%% SSIM %%%%
        
        end
    end
    
%%%% pick the smalllest A as the most similar one %%%%
    [M,I] = min(A(:));
    [I_row, I_col] = ind2sub(size(A),I);
    
    while I_row>179 || I_col>179
       A(I_row,I_col)= A(I_row,I_col) + 100;
       [M,I] = min(A(:));
       [I_row, I_col] = ind2sub(size(A),I);
    end
    I_row
    I_col
%%%% give the next colum/row value %%%%
    blured_resize(ii+21,jj:jj+20)=sample(I_row+21, I_col:I_col+20);
    blured_resize(ii:ii+20,jj+21)=sample(I_row:I_row+20,I_col+21);
    jj
    end
    ii
    figure; 
    subplot(2,1,1); 
    imshow(blured_resize); 
    subplot(2,1,2); 
    imshow(sample); 
end
end
% figure;
% imshow(blur_resize);