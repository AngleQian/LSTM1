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

void MemoryBlock::forwardpass(const std::vector<double> & inputs){
    double inputNet = 0;
    double forgetNet = 0;
    double outputNet = 0;
    
    for(int i = 0; i != inputs.size(); ++i){
        inputNet += inputs[i] * inputGateWeights[i];
        forgetNet += inputs[i] * forgetGateWeights[i];
        outputNet += inputs[i] * outputGateWeights[i];
    }
    
    inputGate = utility::f(inputNet);
    forgetGate = utility::f(forgetNet);
    outputGate = utility::f(outputNet);
    
    for(std::shared_ptr<MemoryCell> memoryCell : memoryCells){
        memoryCell->forwardpass(inputs, inputGate, forgetGate, outputGate);
    }
}

std::vector<double>* MemoryBlock::getOutput() const{
    std::vector<double>* output = new std::vector<double>();
    for(std::shared_ptr<MemoryCell> memoryCell : memoryCells){
        output->push_back(memoryCell->getOutput());
    }
    return output;
}

void MemoryBlock::printUnit(){
    std::cout << "Wf: ";
    utility::printVector(inputGateWeights);
    std::cout << "; Wf: ";
    utility::printVector(forgetGateWeights);
    std::cout << "; Wo: ";
    utility::printVector(outputGateWeights);
    
    std::cout << std::endl;
    
    for(int i = 0; i != memoryCells.size(); ++i){
        std::cout << "Cell: " << i+1 << std::endl;
        memoryCells[i]->printCell();
        std::cout << std::endl;
    }
}
