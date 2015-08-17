function [H HP Hc HPc] = tirbm_sample_multrand2(poshidexp, spacing)
% exploit that the last column is zero
% poshidexp is 3d array
poshidprobs = exp(poshidexp);
poshidprobs_mult = zeros(spacing^2, size(poshidprobs,1)*size(poshidprobs,2)*size(poshidprobs,3)/spacing^2);
% poshidprobs_mult(end,:) = 1;
% TODO: replace this with more realistic activation, bases..
for c=1:spacing
    for r=1:spacing
        temp = poshidprobs(r:spacing:end, c:spacing:end, :);
        poshidprobs_mult((c-1)*spacing+r,:) = temp(:);
    end
end

[S1 P1] = multrand2_fast(poshidprobs_mult');
S = S1';
P = P1';
clear S1 P1

% convert back to original sized matrix
H = zeros(size(poshidexp));
HP = zeros(size(poshidexp));
for c=1:spacing
    for r=1:spacing
        H(r:spacing:end, c:spacing:end, :) = reshape(S((c-1)*spacing+r,:), [size(H,1)/spacing, size(H,2)/spacing, size(H,3)]);
        HP(r:spacing:end, c:spacing:end, :) = reshape(P((c-1)*spacing+r,:), [size(H,1)/spacing, size(H,2)/spacing, size(H,3)]);
    end
end

if nargout >2
    Sc = sum(S);
    Pc = sum(P);
    Hc = reshape(Sc, [size(poshidexp,1)/spacing,size(poshidexp,2)/spacing,size(poshidexp,3)]);
    HPc = reshape(Pc, [size(poshidexp,1)/spacing,size(poshidexp,2)/spacing,size(poshidexp,3)]);
end

return


function [S P] = multrand2_fast(P)
% P is 2-d matrix: 2nd dimension is # of choices

%%
% sumP = row_sum(P); 
% sumP = sum(P,2);
P = P./repmat(sum(P,2)+1, [1,size(P,2)]);

cumP = cumsum(P,2);
% rand(size(P));
unifrnd = rand(size(P,1),1);
% S = cumP > repmat(unifrnd,[1,size(P,2)]);
S = bsxfun(@gt, cumP, unifrnd);
Sindx = diff(S,1,2);
S(:,2:end) = Sindx;

S = double(S);

% cumP = cumsum(P,2);
% % rand(size(P));
% unifrnd = rand(size(P,1),1);
% temp = cumP > repmat(unifrnd,[1,size(P,2)]);
% Sindx = diff(temp,1,2);
% S = zeros(size(P));
% S(:,1) = temp(:, 1);
% S(:,2:end) = Sindx;

return
