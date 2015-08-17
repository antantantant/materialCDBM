function y = conv2_mult(a, B, convopt)
y = [];
if size(a,3)>1
    for i=1:size(a,3)
        y(:,:,i) = conv2(a(:,:,i), B, convopt);
    end
else
    for i=1:size(B,3)
        y(:,:,i) = conv2(a, B(:,:,i), convopt);
    end
end
return
