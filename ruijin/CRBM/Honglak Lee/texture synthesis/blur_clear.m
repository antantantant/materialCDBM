a=imread('blure.jpg');
b=imread('clear.jpg');
a1=a(:,:,1); b1=b(:,:,1);
a_blur=a1(33:374,120:461);b_clear=b1(33:374,120:461);
c=b_clear-a_blur;
% for i = 4:338
%     for j = 4:338
%         if c(i,j)>49
%             a_blur(i-3:i+3,j-3:j+3)=c(i-3:i+3,j-3:j+3)+100;
%         end
%     end
% end
% for j = 2:341
%     for i = 2:341
%         n = 0;
%         while c(i+n,j)>0
%             n=n+1;
%         end
%         m = 0;
%         while c(i-m,j)>0
%             m = m+1;
%         end
%         
%         if m+n>3
%             l = 0;
%             while a_blur(i+l,j-1)<110 & i+l < 342
%                 l = l + 1;
%             end
%             
%             p = 0;
%             while a_blur(i-p,j-1)<110 & i-p > 1
%                 p = p + 1;
%             end
%             
%             % l and p are the strock track for last column
%             % pick the smaller one
%             if l<p
%                 a_blur(i-m:i+n,j) = a_blur(i-m+l:i+n+l,j-1);
%             else 
%                 a_blur(i-m:i+n,j) = a_blur(i-m-p:i+n-p,j-1);
%             end
%         end
%     end
% end

                
        