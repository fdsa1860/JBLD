function [features, action_labels, tr_te_splits] = parseSubJHMDB

dataPath = fullfile('~','research','data','JHMDB');
jointPath = fullfile(dataPath, 'joint_positions');
splitsPath = fullfile(dataPath, 'splits');

opt = struct('T',7,'s',3);

maxVideos = 1000;
features = cell(1, maxVideos);
action_labels = zeros(maxVideos, 1);
tr_te_splits = zeros(3, maxVideos);
count = 1;

actList = listFolder(jointPath);
for i = 1:length(actList)
    fprintf('feature generating %d/%d\n',i,length(actList));
    subDataPath = fullfile(jointPath, actList{i});
    vidList = listFolder(subDataPath);
    for j = 1:length(vidList)
        subsubDataPath = fullfile(subDataPath, vidList{j});
        A = load(fullfile(subsubDataPath, 'joint_positions.mat'),'pos_world');
        
        % compute normalize joints
        norm_positions = positions_to_normalizepositions(A.pos_world);
        
%         % compute relations: distance between two joints
        dist_relations = positions_to_dist_relations(A.pos_world);
%         
%         % compute relations:angle spanned by three joints
        angle_relations = positions_to_angle_relations(A.pos_world);
%         
%         % compute relations: orientation between two joints
        ort_relations = positions_to_ort_relations(A.pos_world);
%         
%         % compute trajectory of positions with cartesian representation:
        cartesian_trajectory = positions_to_cartesian_trajectory(A.pos_world,opt);
%         
%         % compute trajectory of positions with radial representation:
        radial_trajectory = positions_to_radial_trajectory(A.pos_world,opt);
%         
%         % compute trajectory of dist_relations
        dist_relation_trajectory = X_to_trajectory(dist_relations,opt);
%         
%         % compute trajectory of angle_relations
        angle_relation_trajectory = X_to_trajectory(angle_relations,opt);
%         
%         % compute trajectory of ort_relations
        ort_relation_trajectory = X_to_trajectory(ort_relations,opt);
        
%         features{count} = positions;
%         features{count} = norm_positions;
%         features{count} = [norm_positions; dist_relations; ort_relations; angle_relations];
%         features{count} = [dist_relations(:,1:2:end); ort_relations(:,1:2:end)];
        features{count}.feat1 = [norm_positions; dist_relations; ...
            angle_relations; ort_relations];
        features{count}.feat2 = cat(2, cartesian_trajectory,  ...
            radial_trajectory, dist_relation_trajectory, ...
            angle_relation_trajectory, ort_relation_trajectory);
        action_labels(count) = i;
        videoName = [vidList{j}, '.avi'];
        
        for k = 1:3
            splitFile = sprintf('%s_test_split%d.txt', actList{i}, k);
            fid = fopen(fullfile(splitsPath, splitFile));
            C = textscan(fid, '%s %f');
            fclose(fid);
            vName = C{1};
            s = C{2};
            tr_te_splits(k, count) = s(strcmp(videoName, vName));
        end
        
        
        count = count + 1;
    end
end

features(count:end) = [];
action_labels(count:end) = [];
tr_te_splits(:, count:end) = [];

end