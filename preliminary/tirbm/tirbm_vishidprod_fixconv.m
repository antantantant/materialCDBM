function vishidprod2 = tirbm_vishidprod_fixconv(imdata, H, ws)

numchannels = size(imdata,3);
numbases = size(H,3);

% tic
selidx1 = size(H,1):-1:1;
selidx2 = size(H,2):-1:1;
vishidprod2 = zeros(ws,ws,numchannels,numbases);

if numchannels==1
    vishidprod2 = conv2_mult(imdata, H(selidx1, selidx2, :), 'valid');
else
    for b=1:numbases
        vishidprod2(:,:,:,b) = conv2_mult(imdata, H(selidx1, selidx2, b), 'valid');
    end
end

vishidprod2 = reshape(vishidprod2, [ws^2, numchannels, numbases]);

return
