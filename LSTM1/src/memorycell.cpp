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
    cellStatePast = 0;
}

void MemoryCell::forwardpass(const std::vector<double> & inputs, double inputGate, double forgetGate, double outputGate){
    double cellStateNet = 0;
    for(int i = 0; i != inputs.size(); ++i){
        cellStateNet += inputs[i] * cellStateWeights[i];
    }
    double cellStateCandidate = utility::g(cellStateNet);

    cellStateCurrent = forgetGate * cellStatePast + inputGate * cellStateCandidate;
    cellStatePast = cellStateCurrent;
    
    output = outputGate * utility::h(cellStateCurrent);
}

void MemoryCell::printCell(){
    std::cout << "Wcell: ";
    utility::printVector(cellStateWeights);
}
