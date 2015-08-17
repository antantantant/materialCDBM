function y = conv2_mult_pairwise(a, B, convopt)
y = [];

assert(size(a,3)==size(B,3));
for i=1:size(a,3)
    y(:,:,i) = conv2(a(:,:,i), B(:,:,i), convopt);
end

return
