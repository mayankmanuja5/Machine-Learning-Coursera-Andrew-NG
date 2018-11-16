function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
m = length(y); % number of training examples
h=X*theta;
h=sigmoid(h);
temp=h;
h=(-y).*log(h)-(1-y).*log(1-h);
h=h/(m);
t=theta.^2;
t=t./(2*m);
t=t.*lambda;
t(1)=0;
t1=sum(t);


% You need to return the following variables correctly 
J = sum(h)+t1;
grad = zeros(size(theta));
temp=temp-y;
n=temp.*X;
n=sum(n);
grad=n';
grad=grad./m;
tt=theta.*lambda;
tt=tt./m;
tt(1)=0;
grad=grad+tt;


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================

end
