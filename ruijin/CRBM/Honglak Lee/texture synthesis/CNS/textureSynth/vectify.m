% [VEC] = vectify(MTX)
% 
% Pack elements of MTX into a column vector.  Same as VEC = MTX(:)

function vec = vectify(mtx)

vec = mtx(:);
