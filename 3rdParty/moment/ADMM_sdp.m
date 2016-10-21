function output = ADMM_sdp(xa,C,sd,sdv,E,Ev,ep_abs,ep_rel,varargin)

% L = C'*x + g(z) + <y,sd*x+sdv-z> + mu/2*||sd*x+sdv-z||^2 + <yE,E*x+Ev> +
%     mu/2*||E*x+Ev||^2
% make all input column vector formation if possible
% sd and sdv should be cell formation
% dbstop if error;
if nargin == 8
    mode = 0;
    disp('Normal ADMM Mode...');
elseif strcmp(varargin{1},'relax')
    mode = 1;
    disp('Relaxation Mode...');
elseif strcmp(varargin{1},'acc')
    mode = 2;
    disp('Accelerated ADMM Mode...');
end

B = [cell2mat(sd);E];
N = numel(sd);
za = cell(N,1);
yEa = zeros(size(E,1),1);
siz = zeros(N,1);
for i = 1:N
    za{i} = zeros(size(sd{i},1),1);
    siz(i) = sqrt(size(sd{i},1));
end
ya = za;
% zb = za;
yb = ya;
yEb = yEa;
xb = xa;
Iter = 30000;
alpha = ones(1,Iter+1);
CPUtime = zeros(1,Iter);
mu = 1;
Mu = zeros(1,Iter);
p_res = zeros(1,Iter);
d_res = zeros(1,Iter);
tic;
R = sparse(length(xa),length(xa));
for i = 1:N
    R = sd{i}'*sd{i}+R;
end
R = E'*E+R;
InvR = R\speye(length(xa));
toc;
beta = 1;
tic;
for iter = 1:Iter
    if mode == 2
        if iter>2 && max([p_res(iter-2),d_res(iter-2)])-max([p_res(iter-1),d_res(iter-1)])<0
            alpha(iter+1) = 1;
            xf = xa;
            yf = ya;
            yEf = yEa;
        else
            alpha(iter+1) = (1+sqrt(1+4*alpha(iter)^2))/2;
            xf = xa+(alpha(iter)-1)/alpha(iter+1)*(xa-xb);
            yf = cellfun(@plus,ya,cellfun(@mtimes,num2cell((alpha(iter)-1)/alpha(iter+1)*ones(N,1)),...
                cellfun(@minus,ya,yb,'UniformOutput',false),'UniformOutput',false),'UniformOutput',false);
            yEf = yEa+(alpha(iter)-1)/alpha(iter+1)*(yEa-yEb);
        end
    else
        if mode == 1
            beta = rand(1)*0.3+1.5;
        end
        xf = xa;
        yf = ya;
        yEf = yEa;
    end
%     zb = za;
    yb = ya;
    yEb = yEa;
    xb = xa;
        
    v = -C;
    for i = 1:N
        ztemp = sd{i}*xf+sdv{i}+1/mu*yf{i};
        [U,D] = eig(reshape(ztemp,siz(i),siz(i)));
        d = diag(D);
        d(d<=0) = 0;
        za{i} = reshape(U*diag(d)*U',[],1);
        v = -sd{i}'*yf{i}+mu*sd{i}'*(beta*za{i}+(1-beta)*(sd{i}*xf+sdv{i}))-mu*sd{i}'*sdv{i}+v;
    end
    v = -E'*yEf-mu*E'*Ev+v;
    xa = 1/mu*InvR*v;
    for i = 1:N
        ya{i} = yf{i}+mu*(sd{i}*xa+sdv{i}-(beta*za{i}+(1-beta)*(sd{i}*xf+sdv{i})));
        p_res(iter) = (sd{i}*xa+sdv{i}-za{i})'*(sd{i}*xa+sdv{i}-za{i})+p_res(iter);
        d_res(iter) = (mu*sd{i}*(xa-xf))'*(mu*sd{i}*(xa-xf))+d_res(iter);
    end
    yEa = yEf+mu*(E*xa+Ev);
    p_res(iter) = (E*xa+Ev)'*(E*xa+Ev)+p_res(iter);
    p_res(iter) = sqrt(p_res(iter));
    d_res(iter) = sqrt(d_res(iter));
    epsilon_pri = sqrt(size(B,1))*ep_abs+ep_rel*max([norm(cell2mat(za)),norm(B*xa),norm([cell2mat(sdv);Ev])]);
    epsilon_dual = sqrt(length(cell2mat(za)))*ep_abs+ep_rel*norm(cell2mat(ya));
    if p_res(iter)<=epsilon_pri && d_res(iter)<=epsilon_dual
        CPUtime(iter) = toc;
        fprintf('%d/%d\t\t%f\t\t%f/%f\t\t%f/%f\t%f\n',iter,Iter,CPUtime(iter),p_res(iter),epsilon_pri,d_res(iter),epsilon_dual,C'*real(xa));
%         keyboard;
        break;
    end
    Mu(iter) = mu;
    if p_res(iter)>10*d_res(iter)
        mu = mu*2;
    end
    if d_res(iter)>10*p_res(iter)
        mu = mu/2;
    end
%     mu = min([mu*1.1,1e6]);
    CPUtime(iter) = toc;
    fprintf('%d/%d\t\t%f\t\t%f/%f\t\t%f/%f\t%f\n',iter,Iter,CPUtime(iter),p_res(iter),epsilon_pri,d_res(iter),epsilon_dual,C'*real(xa));
%     if iter>=1000 && mod(iter,500)==0
%         keyboard;
%     end
end
output = real(xa);
% save('ADMM_junk_LaLaLa');