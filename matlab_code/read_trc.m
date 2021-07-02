fileName = 'Recording_FX01_day_01_2-Unnamed.trc'
fid = fopen(fileName);
Rows = textscan(fid,'%s','delimiter','\n');

Rows2 = Rows{1}(7:end);
F = cell(1,length(Rows2));
for i = 1:length(Rows2)
    try
    tempRow = Rows2{i};
    t = textscan(tempRow,'%f','delimiter','\t','CollectOutput',1,'treatAsEmpty',{'VOID'});
    lt = length(t{1})-2;
    t2 = reshape(t{1}(3:end),[3 lt/3])';
    fid = find(~isnan(t2(:,1)));
    f2 = t2(fid,:);
    F{i} = f2;
    catch
    end
end

pixdM = 10; 
tic
[Fmerged, mergeN] = mergeTRC(F,pixdM);
toc
