function SurroundSerialNum = GridSCAN_decodeSurround(CenterSerialNum,num_bits)
    HORIZON_DECODE = uint16(21845);
    VERTICAL_DECODE = uint16(43690);
    SurroundSerialNum(3+1) = bitand(bitor(bitand((bitand(CenterSerialNum , HORIZON_DECODE) - 1) , HORIZON_DECODE) , bitand(CenterSerialNum , VERTICAL_DECODE)) , bitshift(uint16(65535) , -(16 - num_bits)));
    SurroundSerialNum(4+1) = bitand(bitor(bitand((bitor(CenterSerialNum , VERTICAL_DECODE) + 1) , HORIZON_DECODE) , bitand(CenterSerialNum , VERTICAL_DECODE)) , bitshift(uint16(65535) , -(16 - num_bits)));
    SurroundSerialNum(1+1) = bitand(bitor(bitand((bitand(CenterSerialNum , VERTICAL_DECODE) - 2) , VERTICAL_DECODE) , bitand(CenterSerialNum , HORIZON_DECODE)) , bitshift(uint16(65535) , -(16 - num_bits)));
    SurroundSerialNum(6+1) = bitand(bitor(bitand((bitor(CenterSerialNum , HORIZON_DECODE) + 2) , VERTICAL_DECODE) , bitand(CenterSerialNum , HORIZON_DECODE)) , bitshift(uint16(65535) , -(16 - num_bits)));
    SurroundSerialNum(0+1) = bitand(bitor(bitand((bitand(SurroundSerialNum(1+1) , HORIZON_DECODE) - 1) , HORIZON_DECODE) , bitand(SurroundSerialNum(1+1) , VERTICAL_DECODE)) , bitshift(uint16(65535) , -(16 - num_bits)));
    SurroundSerialNum(2+1) = bitand(bitor(bitand((bitor(SurroundSerialNum(1+1) , VERTICAL_DECODE) + 1) , HORIZON_DECODE) , bitand(SurroundSerialNum(1+1) , VERTICAL_DECODE)) , bitshift(uint16(65535) , -(16 - num_bits)));
    SurroundSerialNum(5+1) = bitand(bitor(bitand((bitand(SurroundSerialNum(6+1) , HORIZON_DECODE) - 1) , HORIZON_DECODE) , bitand(SurroundSerialNum(6+1) , VERTICAL_DECODE)) , bitshift(uint16(65535) , -(16 - num_bits)));
    SurroundSerialNum(7+1) = bitand(bitor(bitand((bitor(SurroundSerialNum(6+1) , VERTICAL_DECODE) + 1) , HORIZON_DECODE) , bitand(SurroundSerialNum(6+1) , VERTICAL_DECODE)) , bitshift(uint16(65535) , -(16 - num_bits)));
end