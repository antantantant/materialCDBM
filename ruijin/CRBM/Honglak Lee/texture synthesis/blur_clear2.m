clc;clear all;
a=imread('blure.jpg');
b=imread('clear.jpg');
a1=a(:,:,1); b1=b(:,:,1);
a_blur=a1(33:374,120:461);b_clear=b1(33:374,120:461);
c=b_clear-a_blur;
c(141,156);
% figure(1);
% imshow(a_blur);
% for n=1:5
% %     c(141,156)
% for i = 1:342
%     for j = 1:342        
%         if c(i,j)<20/n
%             c(i,j) = 0;
%         end
%         
% %         a_blur(i,j)=a_blur(i,j)+c(i,j);
%         
%         if a_blur(i,j)<190 && a_blur(i,j) + c(i,j)/2 > 199
%             a_blur(i,j) = 190;
%         elseif a_blur(i,j)>200
%             a_blur(i,j) = a_blur(i,j);
%         elseif a_blur(i,j) + c(i,j)/2 < 199
%             a_blur(i,j)=a_blur(i,j)+c(i,j)/2;
%         end      
% 
%     end
%     
% end
% % a_blur=a_blur+c/2;
% c = b_clear - a_blur;  
% 
% end
% % for n = 1:2
% % a_blur = imsharpen(a_blur);
% % end
% figure(2);
% imshow(a_blur);
% for n = 1:3
% a_blur = imsharpen(a_blur);
% end
% figure(3);
% imshow(a_blur);


% for n = 1:10
% %     n
% for i = i:342
%     for j = 1:342
% %         if c(i,j)<10
% %             c(i,j)=0;
% %         else c(i,j)=c(i,j);
% %         end
%         a_blur(i,j)=a_blur(i,j)+c(i,j);
% %         if a_blur(i,j) + c(i,j)/2 > 200
% %             a_blur(i,j) = 199;
% %         else
% %             a_blur(i,j)=a_blur(i,j)+c(i,j);
% %         end      
%     end
% end
% c = b_clear - a_blur;
% k=c(141:143, 155:158);
% a_blur(130,156)
% % figure(n);
% % imshow(a_blur);
% end
% % c(141,156)
% 
% % a_blur=a_blur+c/2;
% for n = 1:3
% a_blur = imsharpen(a_blur);
% end
% figure(2);
% imshow(a_blur);