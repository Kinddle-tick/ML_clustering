function [SerialNum,gridcnt,gridcell] = GridSCAN_encodeALL(Xpos,Ypos,Zpos,Ybottom,epsilon,num_bits)
    gridcnt = zeros(1,64*64);
    gridcell = cell(1,64*64);
    SerialNum = zeros(1,length(Xpos));
    for i = 1:length(Xpos)
        SerialNum(i) = GridSCAN_encode(Xpos(i),Ypos(i),Ybottom,epsilon,num_bits);
        if SerialNum(i)~= 0
            gridcnt(SerialNum(i)) = gridcnt(SerialNum(i))+1;
            gridcell{SerialNum(i)} = [gridcell{SerialNum(i)} ; [Xpos(i) Ypos(i) Zpos(i)]];
        end
    end

end