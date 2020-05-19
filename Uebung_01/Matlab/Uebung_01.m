%% Aufgabe 1b
A = [1, 2, 3; 1, 4, 6; 0, 0, 1];
A_inv = inv(A);
A_adjoint = A_inv*det(A)

A_alternative =  [1, 2, 3; 1, 4, 6; 1, 0, 0];
inv(A_alternative)
%% Aufgabe 2a) 2
X = linspace(1, 6, 6);
A_stat = [4, 1, 6, 2, 1, 4];
B_stat = [5, 6, 1, 1, 4, 1];
C_stat = [3, 3, 4, 2, 3, 3];

A_stat_prob =  A_stat/sum(A_stat);
A_stat_exp = sum(X.*A_stat_prob);
A_stat_exp_ = sum(X.*A_stat)/sum(A_stat);
% function 1 unbiais
A_stat_var_unbiased = 1/(sum(A_stat)-1)*(sum((X-A_stat_exp).^2.*A_stat));
% function 2
A_stat_var_biased = sum(A_stat_prob.*(X-A_stat_exp).^2);

B_stat_prob =  B_stat/sum(B_stat);
B_stat_exp = sum(X.*B_stat_prob);
B_stat_exp_ = sum(X.*B_stat)/sum(B_stat);
% function 1 unbiais
B_stat_var_unbiased = 1/(sum(B_stat)-1)*(sum((X-B_stat_exp).^2.*B_stat));
% function 2
B_stat_var_biased = sum(B_stat_prob.*(X-B_stat_exp).^2);
    
C_stat_prob =  C_stat/sum(C_stat);
C_stat_exp = sum(X.*C_stat_prob);
C_stat_exp_ = sum(X.*C_stat)/sum(C_stat)
% function 1 unbiais
C_stat_var_unbiased = 1/(sum(C_stat)-1)*(sum((X-C_stat_exp).^2.*C_stat))
% function 2
C_stat_var_biased = sum(C_stat_prob.*(X-C_stat_exp).^2);

%%  Aufgabe 2a) 3
KL_A_stat = sum(A_stat_prob.*log(A_stat_prob/(1/6)))
KL_B_stat = sum(B_stat_prob.*log(B_stat_prob/(1/6)))
KL_C_stat = sum(C_stat_prob.*log(C_stat_prob/(1/6)))

%% Aufgabe 2b) 4
P_BA = 0.3;
P_A = 0.03;
P_B = 0.3*0.03+0.1*0.97
P_AB = P_BA*P_A/P_B


%% Aufgabe 3a)
p_si = [0.02, 0.67, 0.23, 0.08];
h_p = -log2(p_si)
H_p = sum(h_p.*p_si)

% max information
p_si_ave = [0.25, 0.25, 0.25, 0.25];
h_p_ave = -log2(p_si_ave)
H_p = sum(h_p_ave.*p_si_ave)

%% Aufgabe 3b)
% 数值解
% Set Initial theta
initial_lambda = zeros(2,1);

% Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 50);

% Run fmincg to obtain the optimal theta
% This function will return theta and the cost
[lambda_such, ~] = fmincg (@(lambda_)(dual_lagrange(lambda_)), ...
    initial_lambda, options);

%% 解析解
syms lambda_1 lambda_2
eqns = [2*exp(1+2*lambda_1+lambda_2)+4*exp(1+4*lambda_1+lambda_2)+6*exp(1+6*lambda_1+lambda_2)+8*exp(1+8*lambda_1+lambda_2)==6;
    exp(1+2*lambda_1+lambda_2)+exp(1+4*lambda_1+lambda_2)+exp(1+6*lambda_1+lambda_2)+exp(1+8*lambda_1+lambda_2) == 1];

[solve_1, solve_2] = solve(eqns, lambda_1, lambda_2)

