function [predicted_labels,W,time] = nn_sos2(X_train, y_train, X_test, opt)

% Nearest Neighbor classification

tStart = tic;

% PCA
nPCA = 3;
[U,Xmean,vars] = pca(X_train);
residuals = cumsum(vars/sum(vars)); plot(residuals,'-.');
[ X_train, Xhat, avsq ] = pcaApply( X_train, U, Xmean, nPCA );

unique_classes = unique(y_train);
n_classes = length(unique_classes);

mu = zeros(size(X_train, 1), n_classes);
Cinv = zeros(size(X_train, 1), size(X_train, 1), n_classes);
Cdet = zeros(1, n_classes);
for ai = 1:n_classes
    X_tmp = X_train(:, y_train==unique_classes(ai));
    mu(:, ai) = mean(X_tmp, 2);
    Xm = bsxfun(@minus, X_tmp, mu(:, ai));
    Cdet(ai) = det(Xm * Xm');
    Cinv(:, :, ai) = inv(Xm * Xm');
end

% if strcmp(opt.metric,'JBLD') || strcmp(opt.metric,'JLD_denoise')
%     HH_center = cell(1, n_classes);
%     %         cparams(1:n_classes) = struct ('prior',0,'alpha',0,'theta',0);
%     M_inv = cell(n_classes, 1);
%     d = cell(n_classes, 1);
%     scale = zeros(n_classes, 1);
%     for ai = 1:n_classes
%         X_tmp = HH_train(y_train==unique_classes(ai));
%         HH_center{ai} = steinMean(cat(3,X_tmp{1:end}));
%         d{ai} = HHdist(X_tmp,HH_center(ai),opt);
%         scale(ai) = mean(d{ai});
% %         ker = exp(-d/scale(ai)/10);
%         M_inv{ai} = getInverseMomentMat(d{ai}, opt.mOrd);
%         %             d(abs(d)<1e-6) = 1e-6;
%         % %             phat = gamfit(d);
%         %             phat = mle(d,'pdf',@gampdf,'start',[1 1],'lowerbound',[0 0],'upperbound',[1.5 inf]);
%         %             cparams(ai).alpha = min(100,phat(1));
%         %             if isinf(cparams(ai).alpha), keyboard;end
%         %             cparams(ai).theta = max(0.01,phat(2));
%         %             cparams(ai).prior = length(X_tmp) / length(X_train);
%         fprintf('processed %d/%d\n',ai,n_classes);
%     end
% end

time.trainTime = toc(tStart);

% test NN
tStart = tic;
[ X_test, Xhat, avsq ] = pcaApply( X_test, U, Xmean, nPCA );

W = zeros(n_classes, size(X_test, 2));
for ai = 1:n_classes
    for j = 1:size(X_test, 2)
        tmp = X_test(:, j) - mu(:, ai);
        W(ai, j) = exp( - tmp' * Cinv(:,:,ai) * tmp) / sqrt(Cdet(ai));
    end
end

[val, ind] = max(W);

% [basis,~] = momentPowers(0, 1, opt.mOrd);
% D2 = zeros(size(D));
% for j = 1:size(D2, 2)
% %     ker = exp(-D2(:, j)./scale/10);
%     for k = 1:size(D2, 1)
%         v = prod( bsxfun( @power, D(k, j), basis), 2);
%         D2(k, j) = v' * M_inv{k} * v;
%     end
% end
% [value, ind] = min(D2);

predicted_labels = unique_classes(ind);

time.testTime = toc(tStart);



end