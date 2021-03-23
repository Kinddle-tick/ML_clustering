function Cluster = GridSCAN(pos,Ybottom,gridscale,num_bits,starter_thre,final_thre)
    MAX_CLUSTER = 30;
    
    [~,gridcnt,gridcell] = GridSCAN_encodeALL(pos.XData,pos.YData,pos.ZData,Ybottom,gridscale,num_bits);
    
    IDX = 1;
    Cluster = cell(1,MAX_CLUSTER);
    
    for i = 1:MAX_CLUSTER
        [val,number] = max(gridcnt);
        if number~=0
            if val>starter_thre
                Cluster{IDX} = [ Cluster{IDX};gridcell{number}];
                gridcnt(number) = 0;
                [gridcnt,Cluster]=GridSCAN_expand(number,IDX,gridcnt,gridcell,starter_thre,num_bits,Cluster);
                if size(Cluster{IDX},1)<final_thre
                    Cluster{IDX} = [];
                else
                    IDX = IDX +1;
                end
            end
        end
    end

end

function [gridcnt,Cluster] = GridSCAN_expand(number,IDX,gridcnt,gridcell,starter_thre,num_bits,Cluster)
    SurroundSerialNum = GridSCAN_decodeSurround(number,num_bits);
    for i =1:8
        if SurroundSerialNum(i)~=0
            if gridcnt(SurroundSerialNum(i))>starter_thre
                Cluster{IDX} = [ Cluster{IDX};gridcell{SurroundSerialNum(i)}];
                gridcnt(SurroundSerialNum(i)) = 0;
                [gridcnt,Cluster]=GridSCAN_expand(SurroundSerialNum(i),IDX,gridcnt,gridcell,starter_thre,num_bits,Cluster);
            else 
                if gridcnt(SurroundSerialNum(i)) ~= 0
                    Cluster{IDX} = [ Cluster{IDX};gridcell{SurroundSerialNum(i)}];
                    gridcnt(SurroundSerialNum(i)) = 0;
                end
            end
        end
    end
end