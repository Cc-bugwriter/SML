function [G, d_G] = dual_lagrange(lambda_)
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明
g = zeros(4,1);
d_g_1 = zeros(4,1);
d_g_2 = zeros(4,1);

for i = 1:length(g)
    g(i) = exp(1+2*i*lambda_(1,:)+lambda_(2,:));
    d_g_1(i) = (2*i)*exp(1+2*i*lambda_(1,:)+lambda_(2,:));
    d_g_2(i) = exp(1+2*i*lambda_(1,:)+lambda_(2,:));
end
G = sum(g) + 6*lambda_(1,:) + lambda_(2,:);
d_G = sum(d_g_1) +6 + sum(d_g_2); 
end

