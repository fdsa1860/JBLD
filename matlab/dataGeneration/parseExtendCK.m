function [seq, emoLabel, subLabel, insLabel] = parseExtendCK

nSeq = 593;
nSub = 123;
nLandmark = 68;
dataPath = fullfile('~','research','data','extendCK');

subLabel = zeros(1, nSeq);
insLabel = zeros(1, nSeq);

aList = dir(fullfile(dataPath,'Landmarks'));
subName = cell(1, nSub);
count = 1;
for i = 1:length(aList)
    if strcmp(aList(i).name, '.') || strcmp(aList(i).name, '..')
        continue;
    end
    if exist(fullfile(dataPath, 'Landmarks', aList(i).name), 'dir')
        subName{count} = aList(i).name;
        count = count + 1;
    end
end

seqName = cell(1, nSeq);
count = 1;
for i = 1:nSub
    bList = dir(fullfile(dataPath, 'Landmarks', subName{i}));
    for j = 1:length(bList)
        if strcmp(bList(j).name, '.') || strcmp(bList(j).name, '..')
            continue;
        end
        if exist(fullfile(dataPath, 'Landmarks', subName{i}, bList(j).name), 'dir')
            seqName{count} = bList(j).name;
            subLabel(count) = i;
            insLabel(count) = j;
            count = count + 1;
        end
    end
end
assert(count-1 == nSeq);

seq = cell(1, nSeq);
for i = 1:nSeq
    filePath = fullfile(dataPath, 'Landmarks', subName{subLabel(i)}, seqName{i});
    fList = dir(fullfile(filePath, '*.txt'));
    seq{i} = zeros(nLandmark*2, length(fList));
    for j = 1:length(fList)
        fr = load(fullfile(filePath, fList(j).name));
        fr = fr';
        seq{i}(:, j) = fr(:);
    end
end

validInd = true(1, nSeq);
emoLabel = zeros(1, nSeq);
for i = 1:nSeq
    EmotionPath = fullfile(dataPath, 'Emotion', subName{subLabel(i)}, seqName{i});
    eList = dir(fullfile(EmotionPath, '*.txt'));
    if isempty(eList)
        validInd(i) = false;
        continue;
    end
    l = load(fullfile(EmotionPath, eList.name));
    emoLabel(i) = l;
end

emoLabel(~validInd) = [];
subLabel(~validInd) = [];
insLabel(~validInd) = [];
seq(~validInd) = [];

end