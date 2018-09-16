//
//  memorycell.cpp
//  LSTM1
//
//  Created by Angle Qian on 12/09/2018.
//  Copyright © 2018 Angle Qian. All rights reserved.
//

#include "../include/memorycell.hpp"

MemoryCell::MemoryCell() {
    cellStateWeights = std::vector<double>();
}

void MemoryCell::printCell(){
    std::cout << "Wcell:";
    utility::printVector(cellStateWeights);
}
