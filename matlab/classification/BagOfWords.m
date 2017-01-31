function [predicted_labels, time] = BagOfWords(X_train, y_train, X_test, opt)

m1 = size(X_train{1}.feat1, 1);
m2 = size(X_train{2}.feat2, 2);
m = m1 + m2;
nCluster = opt.nCluster;
nTrain = length(X_train);
X_train_bow = zeros(m * nCluster, nTrain);
nTest = length(X_test);
X_test_bow = zeros(m * nCluster, nTest);

rng(0);

for mi = 1:m
    
    if mi <= m1, d = 1; else d = 2; end
    
    % pool all features
    p = randperm(nTrain);
    maxSize = 25000;
    
    feat = zeros(d, maxSize);
    count = 1;
    for i = 1:nTrain
        X_curr = X_train{p(i)};
        if mi <= m1
            f_curr = X_curr.feat1(mi, :);
        else
            f_curr = X_curr.feat2(:, mi-m1, :);
            f_curr = reshape(f_curr, 2, []);
        end
            
        n = size(f_curr, 2);
        feat(:, count:count+n-1) = f_curr;
        count = count + n;
        if count >= maxSize, break; end
    end
    count = min(count, maxSize);
    feat(:, count:end) = [];
    
    % cluster the features into k clusters
    [~, dict] = kmeans(feat', nCluster, 'Replicates', 8);
    
    % get BOW representation of training data
    for i = 1:nTrain
        X_curr = X_train{i};
        if mi <= m1
            f_curr = X_curr.feat1(mi, :);
        else
            f_curr = X_curr.feat2(:, mi-m1, :);
            f_curr = reshape(f_curr, 2, []);
        end
        D2 = pdist2(dict, f_curr');
        [~, ind] = min(D2);
        h = hist(ind, 1:nCluster);
        h = h / norm(h);
        X_train_bow((mi-1)*nCluster+1:mi*nCluster, i) = h';
    end
    
    % get VLAD representation of testing data
    for i = 1:nTest
        X_curr = X_test{i};
        if mi <= m1
            f_curr = X_curr.feat1(mi, :);
        else
            f_curr = X_curr.feat2(:, mi-m1, :);
            f_curr = reshape(f_curr, 2, []);
        end
        D2 = pdist2(dict, f_curr');
        [~, ind] = min(D2);
        h = hist(ind, 1:nCluster);
        h = h / norm(h);
        X_test_bow((mi-1)*nCluster+1:mi*nCluster, i) = h';
    end
    
end

% SVM
[predicted_labels] = svm_one_vs_all_predict(X_train_bow, y_train, X_test_bow, opt.C_val);

time.trainTime = 0;
time.testTime = 0;

end