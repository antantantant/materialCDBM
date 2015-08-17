% sample the pooling layer
function hidsample = samplepooling_v1(images_all, sample_size, ws, spacing, model)
    hidsample = [];
    hbias_vec = model.hbias_vec;
    pars = model.pars;
    
    for imgid = 1:numel(images_all)
        imdata = images_all{imgid}+0; % convert to double if logical
        imdata = trim_image_for_spacing_fixconv(imdata, ws, spacing);
        for i = 1:sample_size
            % get hidden layer activation (expected values)
            poshidexp = crbm_inference(imdata, W, hbias_vec, pars);
            % === the following section draw hidden layer realizations 
            % (within each spacing*spacing segment, at most one node can be active)
            poshidprobs = exp(poshidexp);
            poshidprobs_mult = zeros(spacing^2+1, size(poshidprobs,1)*size(poshidprobs,2)*size(poshidprobs,3)/spacing^2);
            poshidprobs_mult(end,:) = 1;
            for c=1:spacing
                for r=1:spacing
                    temp = poshidprobs(r:spacing:end, c:spacing:end, :);
                    poshidprobs_mult((c-1)*spacing+r,:) = temp(:);
                end
            end
            S = multrand2(poshidprobs_mult')';
            % convert to pooling sized matrix
            % dim 1 - height, dim 2 - width, dim 3 - basis
            H = reshape(sum(S(1:spacing^2,:),1)>0,...
                [size(poshidexp,1)/spacing, size(poshidexp,2)/spacing, size(poshidexp,3)]);
            % === DONE drawing
            hidsample{length(hidsample)+1} = H;
        end
    end