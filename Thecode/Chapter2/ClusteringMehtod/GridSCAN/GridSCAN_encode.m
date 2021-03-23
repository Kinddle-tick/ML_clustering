function SerialNum = GridSCAN_encode(Xpos,Ypos,Ybottom,epsilon,num_bits)
    Ystandard = Ybottom + epsilon*2^(num_bits/2-1);
    Xstandard = 0;
    SerialNum = uint16(0);
    step = epsilon * 2^(num_bits / 2-1);
    
    for i = 1:num_bits/2
        step = step/2;
        if Xpos>Xstandard   
            if Ypos>Ystandard  
                SerialNum = bitor(bitshift(uint16(SerialNum),2),uint16(2));
                Ystandard = Ystandard +step;
            else
                SerialNum = bitor(bitshift(uint16(SerialNum),2),uint16(0));
                Ystandard = Ystandard-step;
            end
            Xstandard = Xstandard+step;       
        else  
            if Ypos>Ystandard        
                SerialNum = bitor(bitshift(uint16(SerialNum),2),uint16(3));
                Ystandard = Ystandard+step;         
            else
                SerialNum = bitor(bitshift(uint16(SerialNum),2),uint16(1));
                Ystandard = Ystandard-step;
            end
            Xstandard = Xstandard-step;
        end
    end
end