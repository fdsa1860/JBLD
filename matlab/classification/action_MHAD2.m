function action_MHAD2(data, tr_info, labels, opt)

% jointsVel = getVelocity(data.joints);
% HH = getHH(jointsVel, opt);
% feat = HH;
% feat = getLogHH(HH);

[preprocessed_data]= Skeleton_Preprocessing(data.joints); 

%Trajectory analysis
options.window_size=30; 
options.remove_mean=0; 
options.step=1; 
options.PC_No=3; 
% options.n=60;
options.n=105;
options.k=3; 
options.t=0.01; 


%compute features 
if ~exist(fullfile('..','expData','VelocityVector.mat'), 'file')
    VelocityVector = VelocityVectorFeature(preprocessed_data, options);
    save(fullfile('..','expData','VelocityVector.mat'), 'VelocityVector');
else
    VelocityVector = importdata(fullfile('..','expData','VelocityVector.mat'));
end
%Read the feature that we computed

%compute the histogram that represent that action
feat = zeros(2*(options.n-options.k), length(VelocityVector));
for i = 1:length(VelocityVector)
    Histogram=getHistogram(VelocityVector{i});
    Histogram = Histogram / norm(Histogram);
    feat(:, i) = Histogram(:);
end

tr_subjects = tr_info.tr_subjects;
te_subjects = tr_info.te_subjects;

subject_labels = labels.subject_labels;
action_labels = labels.action_labels;

tr_ind = ismember(subject_labels, tr_subjects);
te_ind = ismember(subject_labels, te_subjects);

X_train = feat(:,tr_ind);
y_train = action_labels(tr_ind);
X_test = feat(:,te_ind);
y_test = action_labels(te_ind);

nAction = length(unique(action_labels));
model = cell(1, nAction);
y = zeros(size(y_train));
for i = 1:nAction
    y(y_train==i) = 1;
    y(y_train~=i) = -1;
    model{i} = svmtrain(y', X_train', sprintf('-s 0 -t 0 -b 1 -w1 %f -w-1 %f -c 10',nnz(y_train~=i), nnz(y_train==i)));
end

W = zeros(nAction, length(y_test));
for k = 1:length(model)
    [val, ~, prob] = svmpredict(y_test', X_test', model{k}, '-b 1');
    W(k,:) = prob(:,1)';
end

[val, ind] = max(W);
predicted_labels = ind;

accuracy = nnz(y_test==predicted_labels)/ length(y_test);
accuracy
unique_classes = unique(y_test);
n_classes = length(unique_classes);
class_wise_accuracy = zeros(1, n_classes);
confusion_matrix = zeros(n_classes, n_classes);
for i = 1:n_classes
    temp = find(y_test == unique_classes(i));
    if ~isempty(temp)
        class_wise_accuracy(i) =...
            nnz(predicted_labels(temp)==unique_classes(i)) / length(temp);
        confusion_matrix(i, :) = ...
            hist(predicted_labels(temp), unique_classes) / length(temp);
    else
        class_wise_accuracy(i) = 1;
        confusion_matrix(i, i) = 1;
    end
end

results_dir = fullfile('..','expData','res');
if ~exist(results_dir,'dir')
    mkdir(results_dir);
end

save ([results_dir, '/classification_results.mat'],...
    'accuracy', 'class_wise_accuracy','confusion_matrix');

end