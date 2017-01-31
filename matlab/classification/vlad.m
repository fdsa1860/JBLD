function [predicted_labels, time] = vlad(X_train, y_train, X_test, opt)

% pool all features
maxSize = 25000;
d = size(X_train{1}, 1);
feat = zeros(d, maxSize);
count = 1;
for i = 1:length(X_train)
    X_curr = X_train{i};
    n = size(X_curr, 2);
    feat(:, count:count+n-1) = X_curr;
    count = count + n;
end
feat(:,count:end) = [];

% cluster the features into k clusters
nCluster = opt.nCluster;
D = pdist2(feat', feat');
W = exp(-D);
NcutDiscrete = ncutW(W, nCluster);
label = sortLabel_order(NcutDiscrete, 1:size(D,1));

% get the dictionary
dict = zeros(d, nCluster);
for i = 1:nCluster
    dict(:, i) = mean(feat(:, label==i), 2);
end

% get BOW VLAD representation of training data
X_train_bow = zeros(nCluster, length(X_train));
for i = 1:length(X_train)
    X_curr = X_train{i};
    D2 = pdist2(dict', X_curr');
    [~, ind] = min(D2);
    T = size(X_curr, 2);
    for t = 1:T
        X_train_bow(ind(t), i) = X_train_bow(ind(t), i) + D2(ind(t), t);
    end
end

% get BOW VLAD representation of testing data
X_test_bow = zeros(nCluster, length(X_test));
for i = 1:length(X_test)
    X_curr = X_test{i};
    D2 = pdist2(dict', X_curr');
    [~, ind] = min(D2);
    T = size(X_curr, 2);
    for t = 1:T
        X_test_bow(ind(t), i) = X_test_bow(ind(t), i) + D2(ind(t), t);
    end
end

% SVM
[predicted_labels] = svm_one_vs_all_predict(X_train_bow, y_train, X_test_bow, opt.C_val);

time.trainTime = 0;
time.testTime = 0;

end