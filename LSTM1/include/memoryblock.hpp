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

class Layer;

class MemoryBlock : public Unit {
public:
    MemoryBlock(int, double, long);
    
    void forwardpass(const std::vector<double>&);
    void backwardpass() {};
    void backwardpass(const std::shared_ptr<Layer>, const std::shared_ptr<Layer>, int);
    void calcDelta(const std::shared_ptr<Layer>, int);
    void calcInternalErrorGatePartials(double*, double*, const std::shared_ptr<Layer>, const std::shared_ptr<Layer>, MemoryBlock*, int, int);
    void applyWeightChanges();
    
    void flushState();
    void printUnit();
    
    std::vector<std::shared_ptr<MemoryCell>> getMemoryCells() { return memoryCells; }
    std::shared_ptr<std::vector<double>> getInputGateWeights() { return inputGateWeights; }
    std::shared_ptr<std::vector<double>> getForgetGateWeights() { return forgetGateWeights; }
    std::shared_ptr<std::vector<double>> getOutputGateWeights() { return outputGateWeights; }
    std::shared_ptr<std::vector<double>> getOutput() const;
    double getDelta() { return delta; }
    double getOutputWeightToCellInPrevLayer(long);
    double getInputNet() { return inputNet; }
    double getInputGate() { return inputGate; }
    double getForgetNet() { return forgetNet; }
    double getForgetGate() { return forgetGate; }
    double getOutputGate() { return outputGate; }
private:
    double alpha;
    std::vector<std::shared_ptr<MemoryCell>> memoryCells;
    
    std::shared_ptr<std::vector<double>> pendingInputGateWeights;
    std::shared_ptr<std::vector<double>> pendingForgetGateWeights;
    std::shared_ptr<std::vector<double>> pendingOutputGateWeights;
    std::shared_ptr<std::vector<double>> inputGateWeights;
    std::shared_ptr<std::vector<double>> forgetGateWeights;
    std::shared_ptr<std::vector<double>> outputGateWeights;
    
    double inputNet;
    double inputGate;
    double forgetNet;
    double forgetGate;
    double outputNet;
    double outputGate;
    
    double delta;
};

#endif /* neuron_hpp */
