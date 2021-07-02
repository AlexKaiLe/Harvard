% addpath('ratData/TRC/')
% addpath('ratData/TRC2/')


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


guessDist = [20.1706 31.6904 37.0356];

Fmn = zeros(size(Fmerged));
for i =1:length(Fmerged)
    Fmn(i) = size(Fmerged{i},1);
end
n = length(Fmerged);
nMarkers = max(Fmn);

markers = nan(n,3,nMarkers);
for i = 1:n
    markers(i,:,1:Fmn(i)) = Fmerged{i}';
end

markersNew = nan(size(markers));

mfmn = unique(Fmn);
templateCell = cell(1,max(Fmn));
for i = mfmn
    templateCell{i} = nchoosek(1:i,3);
end


tic % 10 min
% find first headcap
checkit = 1; nidx = 1; threshd = 1; threshp = 2;
while checkit
    fi = Fmerged{nidx};
    ni1 = length(fi);
    v1 = templateCell{ni1};
    dA = zeros(size(v1,1),3); dD = zeros(size(v1,1),1);
    for i = 1:size(v1,1)
        dt1 = pdist2(fi(v1(i,1),:),fi(v1(i,2),:));
        dt2 = pdist2(fi(v1(i,2),:),fi(v1(i,3),:));
        dt3 = pdist2(fi(v1(i,1),:),fi(v1(i,3),:));
        dA(i,:) = sort([dt1 dt2 dt3]);
        dD(i) = sum(abs(dA(i,:)-guessDist));
    end
    [ld di] = min(dD);
    trips = v1(di,:);
    mids = mean(fi(v1(di,:),:));
    dds = ld;
    
    if ld<threshd
        checkit = 0;
    else
        nidx = nidx+1;
    end
end

startidx = nidx; 


checkidx = 1:300:n;
nd = length(checkidx);
trip = nan(n,3); dd = nan(n,1); mid = nan(n,3); point = nan(n,3,3);
trip(startidx,:) = trips; dd(startidx) = dds; mid(startidx,:) = mids; point(startidx,:,:) = points;
checkit = 0;
for ni = startidx+1:(300*240)
    if rem(ni,18000)==0
        fprintf(1,['Processed points for t = ' num2str(ni)  '\n']);
    end
    fi = Fmerged{ni};
    pointi = squeeze(point(ni-1,:,:));
    [kn kd] = knnsearch(fi,pointi);
    if sum(kd)<threshp
        trip(ni,:) = kn;
        % dd(ni) = sum(kd);
        mid(ni,:) = mean(fi(kn,:));
        point(ni,:,:) = fi(kn,:);
        checkit = 0;
    else
        checkit = 1;
    end
    if ismember(ni,checkidx) || checkit
        ni
        fi = Fmerged{ni};
        ni1 = length(fi);
        v1 = templateCell{ni1};
        dA = zeros(size(v1,1),3); dD = zeros(size(v1,1),1);
        for i = 1:size(v1,1)
            dt1 = pdist2(fi(v1(i,1),:),fi(v1(i,2),:));
            dt2 = pdist2(fi(v1(i,2),:),fi(v1(i,3),:));
            dt3 = pdist2(fi(v1(i,1),:),fi(v1(i,3),:));
            dA(i,:) = sort([dt1 dt2 dt3]);
            dD(i) = sum(abs(dA(i,:)-guessDist));
        end
        [ld di] = min(dD);
        trip(ni,:) = v1(di,:);
        mid(ni,:) = mean(fi(v1(di,:),:));
        dd(ni) = ld;
        if ld<threshd
            checkit = 0;
            point(ni,:,:) = fi(v1(di,:),:);
        end
    end
end
toc








pixd = 2; memd = 10; 
% LINKING MARKERS IN TIME
n = length(Fmerged);
cSize = 27000;
nstarts = 1:cSize:n;
nends = nstarts+cSize-1;
if nends(end)>n
    nends(end)=n;
end
ns = length(nstarts);

Ppar = cell(1,ns);
Flpar = cell(1,ns);
Mpar = cell(1,ns);
MFpar = cell(1,ns);
Pleft = cell(1,ns);

tic
for i = 1:length(nstarts)
idxi = nstarts(i):nends(i);
fprintf(1,['Processing point ' num2str(idxi(1)) '\n']);
F2 = Fmerged(idxi);
[P,~,Fl,~] = linkTRC(F2,pixd,memd);
Ppar{i} = P;
Flpar{i} = Fl;
end
toc








[markersB,~,PleftB] = arrangePB(Pbig);



guessDist = [20.1706 31.6904 37.0356];

Fmn = zeros(size(Fmerged));
for i =1:length(Fmerged)
    Fmn(i) = size(Fmerged{i},1);
end
n = length(Fmerged);
nMarkers = max(Fmn);

markers = nan(n,3,nMarkers);
for i = 1:n
    markers(i,:,1:Fmn(i)) = Fmerged{i}';
end

markersNew = nan(size(markers));

mfmn = unique(Fmn);
templateCell = cell(1,max(Fmn));
for i = mfmn
    templateCell{i} = nchoosek(1:i,3);
end
tic % 10 min
checkidx = 1:60:n;
nd = length(checkidx);
trip = zeros(nd,3); dd = zeros(nd,1); mid = zeros(nd,3);
for ni = 1:nd
    nidx = checkidx(ni);
    if rem(ni,100)==0
        fprintf(1,['Processed points for t = ' num2str(ni) ' of ' num2str(nd) '\n']);
    end
    fi = Fmerged{nidx};
    ni1 = length(fi);
    v1 = templateCell{ni1};
    dA = zeros(size(v1,1),3); dD = zeros(size(v1,1),1);
    for i = 1:size(v1,1)
        dt1 = pdist2(fi(v1(i,1),:),fi(v1(i,2),:));
        dt2 = pdist2(fi(v1(i,2),:),fi(v1(i,3),:));
        dt3 = pdist2(fi(v1(i,1),:),fi(v1(i,3),:));
        dA(i,:) = sort([dt1 dt2 dt3]);
        dD(i) = sum(abs(dA(i,:)-guessDist));
    end
    [ld di] = min(dD);
    trip(ni,:) = v1(di,:);
    mid(ni,:) = mean(fi(v1(di,:),:));
    dd(ni) = ld;
end
toc

% shake markers
tic
pixd = 2;
markersNew(i,:,:) = squeeze(markers(1,:,:)); m1 = squeeze(markers(1,:,:));
m1(isnan(m1)) = inf;
dd = nan(nMarkers,n);
for i = 2:n
    if rem(i,60*300)==0
        fprintf(1,['Processing points for t = ' num2str(i/300) '\n']);
    end
    m1 = squeeze(markersNew(i-1,:,:))';
    mn = zeros(nMarkers,3);
    m2 = squeeze(markers(i,:,:))'; m2(isnan(m2))=inf;
    [id d2] = knnsearch(m2,m1);
    
    mi = 1:nMarkers;
    mmat = [mi' id d2];
    mmatc = mmat(d2<pixd,:);
    
    oldid = find(d2<pixd);
    newid = id(oldid);
    
    [c1 ia1 ic1] = unique(mmatc(:,2));
    mmatc2 = mmatc(ia1,:);
    left1 = setdiff(1:nMarkers,mmatc2(:,1));
    left2 = setdiff(1:nMarkers,mmatc2(:,2));
    
    mn(mmatc2(:,2),:) = m2(mmatc2(:,1),:); 
    mn(left2,:) = m2(left1,:);
    markersNew(i,:,:) = mn';
end
toc

dd = zeros(n-1,nMarkers);
for i = 1:nMarkers
    tm = markersNew(:,:,i);
    dtm = diff(tm);
    dtm2 = sqrt(sum(dtm.^2,2));
    dd(:,i) = dtm2;
end

% expand to double with gaps
pixd = 5;
nm2 = nan(n,3,nMarkers*2);
for i = 1:nMarkers
    i
    dt = dd(:,i);
    tm = markersNew(:,:,i);
    ct = dt<pixd;
    cti = find(ct==0);
    
    if cti(end) ~= n
        cti = [cti; n];
    end
    
   lc = length(cti);
   lc
   chunks = [1 cti(1)];
   for l = 2:lc
       chunks = [chunks; cti(l-1)+1 cti(l)];
   end
   
   chodd = 1:2:length(chunks);
   che = 2:2:length(chunks);
    
   for c = chodd
      nm2(chunks(c,1):chunks(c,2),:,i) = tm(chunks(c,1):chunks(c,2),:);
   end
   for c = che
      nm2(chunks(c,1):chunks(c,2),:,i+nMarkers) = tm(chunks(c,1):chunks(c,2),:);
   end      
end



    
imagesc(reshape(markersNew(:,:,:),[size(markersNew,1) size(markersNew,3)*3])')


startp = min(checkidx(dd<.5));
m1 = Fm{startp};
for i = strtp:n
    
    
end
    





pixd = 1; memd = 20;
[P, F3, Fl, Ftemp] = linkTRC(F,pixd,memd);


scatter3(point(1:10000,1,1),point(1:10000,2,1),point(1:10000,3,1),[],'k','.')



