function [Fmerged, mergeN] = mergeTRC(F,pixd)

Fmerged = cell(size(F));
mergeN = []; mergeP = [];

l2 = length(F);
for i = 1:l2
    
    if rem(i,60*300)==0
        fprintf(1,['Processing points for t = ' num2str(i/300) '\n']);
    end
    
    ft = F{i};
    
    if ~isempty(ft)
        fn = length(ft);
        d = pdist2(ft,ft);
        
        [ds dsi] = sort(d,2);
        
        if length(ds)>5
            % closest 5 points within reason
            fi = dsi(find(ds(:,5)>200));
            if ~isempty(fi)
                ft(fi,:) = [];
                fn = length(ft);
                d = pdist2(ft,ft);
            end
            
            dn = nan(size(d));
            d2 = tril(dn)+triu(d,1);
            
            [mi,mj] = find(d2<pixd);
            
                
            while ~isempty(mi)
                np = unique([mi mj]); picki = zeros(length(np),3); di = zeros(length(np),1);
                
                if i == 1
                    maxd = np(1);
                elseif ~isempty(Fmerged{i-1})
                    for ci = 1:length(np)
                        picki(ci,:) = ft(np(ci),:); di(ci) = min(pdist2(Fmerged{i-1},ft(np(ci),:)));
                    end
                    maxd = np(find(di==max(di)));
                else
                    np = unique([mi mj]); picki = zeros(length(np),3); di = zeros(length(np),1);
                    maxd = np(1);
                end
                ftt = ft;
                mergeN = [mergeN i]; mergeP = [mergeP; ftt(maxd,:)];
                ftt(maxd,:) = [];
                newd = pdist2(ftt,ftt);
                dn = nan(size(newd));
                newd2 = tril(dn)+triu(newd,1);
                [mi mj] = find(newd2<pixd);
                ft = ftt;
            end
            Fmerged{i} = ft;
        else
            Fmerged{i} = ft;
        end
    end
end




