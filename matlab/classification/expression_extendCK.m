function expression_extendCK(data, tr_info, labels, opt)

jointsVel = getVelocity(data.joints);
% HH = getHH(jointsVel, opt);
% feat = HH;
feat = jointsVel;

total_preprocessingTime = toc(opt.tStart);

n_tr_te_splits = size(tr_info.tr_subjects, 1);
tr_subjects = tr_info.tr_subjects;
te_subjects = tr_info.te_subjects;

subject_labels = labels.subject_labels;
action_labels = labels.action_labels;

results_dir = fullfile('..','expData','res');
if ~exist(results_dir,'dir')
    mkdir(results_dir);
end

trainTime = zeros(n_tr_te_splits, 1);
testTime = zeros(n_tr_te_splits, 1);
predicted_labels = zeros(size(action_labels));
for si = 1:n_tr_te_splits
    fprintf('Processing %d/%d ...\n',si,n_tr_te_splits);
    
    tr_subject_ind = ismember(subject_labels, tr_subjects(si,:));
    te_subject_ind = ismember(subject_labels, te_subjects(si,:));
    tr_ind = tr_subject_ind;
    te_ind = te_subject_ind;
    
    X_train = feat(tr_ind);
    nTrain = length(X_train);
    y_train = action_labels(tr_ind);
    X_test = feat(te_ind);
    nTest = length(X_test);
    y_test = action_labels(te_ind);
    
    % train NN
    [y_pred,~,time] = nn(X_train, y_train, X_test, opt);
    
    predicted_labels(te_ind) = y_pred;
    
    trainTime(si) = time.trainTime;
    testTime(si) = time.testTime;

end

total_accuracy = nnz(predicted_labels==action_labels)/ length(action_labels);

unique_classes = unique(action_labels);
n_classes = length(unique_classes);
class_wise_accuracy = zeros(1, n_classes);
confusion_matrix = zeros(n_classes, n_classes);
for i = 1:n_classes
    temp = find(action_labels == unique_classes(i));
    if ~isempty(temp)
        class_wise_accuracy(i) =...
            nnz(predicted_labels(temp)==unique_classes(i)) / length(temp);
        confusion_matrix(i, :) = ...
            hist(predicted_labels(temp), unique_classes) / length(temp);
    else
        class_wise_accuracy(i) = -1;
        confusion_matrix(i, i) = -1;
    end
end

total_trainTime = sum(trainTime);
total_testTime = sum(testTime);
total_runtime = toc(opt.tStart);

save ([results_dir, '/classification_results', '.mat'],...
    'total_accuracy', 'class_wise_accuracy', 'confusion_matrix',...
    'total_trainTime','total_testTime','total_preprocessingTime',...
    'total_runtime','predicted_labels','action_labels');

end