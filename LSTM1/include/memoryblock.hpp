//
//  neuron.hpp
//  LSTM1
//
//  Created by Angle Qian on 11/09/2018.
//  Copyright © 2018 Angle Qian. All rights reserved.
//

#ifndef neuron_hpp
#define neuron_hpp

#include "memorycell.hpp"
#include "unit.hpp"

class MemoryBlock : public Unit {
public:
    MemoryBlock(int);
    
    void forwardpass(const std::vector<double>&);
    
    void printUnit();
    
    std::vector<std::shared_ptr<MemoryCell>>* getMemoryCells() { return &memoryCells; }
    std::vector<double>* getInputGateWeights() { return &inputGateWeights; }
    std::vector<double>* getForgetGateWeights() { return &forgetGateWeights; }
    std::vector<double>* getOutputGateWeights() { return &outputGateWeights; }
    std::vector<double>* getOutput() const;
private:
    std::vector<std::shared_ptr<MemoryCell>> memoryCells;
    
    std::vector<double> inputGateWeights;
    std::vector<double> forgetGateWeights;
    std::vector<double> outputGateWeights;
    
    double inputGate;
    double forgetGate;
    double outputGate;
};

#endif /* neuron_hpp */
