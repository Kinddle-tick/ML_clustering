% Description: 
% Author: Yu Xuyao
% Email: yxy19991102@163.com
% Date: 2020-12-25 00:26:00
% LastEditTime: 2020-12-25 00:26:00
% LastEditors: Yu Xuyao
function [IDX, isnoise]=DBSCAN(X,epsilon,MinPts)
    C=0;
    n=size(X,1);
    IDX=zeros(n,1);
    D=pdist2(X,X);
    visited=false(n,1);
    isnoise=false(n,1);
    for i=1:n
        if ~visited(i)
            visited(i)=true;
            
            Neighbors=RegionQuery(i,D,epsilon);
            if numel(Neighbors)<MinPts
                % X(i,:) is NOISE
                isnoise(i)=true;
            else
                C=C+1;
                [IDX,visited] = ExpandCluster(i,Neighbors,D,C,IDX,visited,MinPts,epsilon);
            end
        end
    end
end

function Neighbors=RegionQuery(i,D,epsilon)
        Neighbors=find(D(i,:)<=epsilon);
end

function [IDX,visited] = ExpandCluster(i,Neighbors,D,C,IDX,visited,MinPts,epsilon)
        IDX(i)=C;
        
        k = 1;
        while true
            j = Neighbors(k);
            
            if ~visited(j)
                visited(j)=true;
                Neighbors2=RegionQuery(j,D,epsilon);
                if numel(Neighbors2)>=MinPts
                    Neighbors=[Neighbors Neighbors2];   %#ok
                end
            end
            if IDX(j)==0
                IDX(j)=C;
            end
            
            k = k + 1;
            if k > numel(Neighbors)
                break;
            end
        end
end

