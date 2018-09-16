//
//  neuron.cpp
//  LSTM1
//
//  Created by Angle Qian on 11/09/2018.
//  Copyright © 2018 Angle Qian. All rights reserved.
//

#include "../include/memoryblock.hpp"

MemoryBlock::MemoryBlock(int cellsPerBlock) {
    for(int i = 0; i != cellsPerBlock; ++i){
        std::shared_ptr<MemoryCell> memoryCell(new MemoryCell());
        memoryCells.push_back(memoryCell);
    }
    inputGateWeights = std::vector<double>();
    forgetGateWeights = std::vector<double>();
    outputGateWeights = std::vector<double>();
}

void MemoryBlock::printUnit(){
    std::cout << "Wf:";
    utility::printVector(inputGateWeights);
    std::cout << "; Wf:";
    utility::printVector(forgetGateWeights);
    std::cout << "; Wo:";
    utility::printVector(outputGateWeights);
    
    std::cout << std::endl;
    
    for(int i = 0; i != memoryCells.size(); ++i){
        std::cout << "Cell: " << i+1 << std::endl;
        memoryCells[i]->printCell();
        std::cout << std::endl;
    }
}
