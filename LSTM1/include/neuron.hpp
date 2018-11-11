//
//  neuron.hpp
//  LSTM1
//
//  Created by Angle Qian on 15/09/2018.
//  Copyright © 2018 Angle Qian. All rights reserved.
//

#ifndef neuron_h
#define neuron_h

#include "unit.hpp"

class Layer;

class Neuron : public Unit {
public:
    Neuron(double);
    
    void forwardpass(double);
    void forwardpass(const std::vector<double>&);
    void backwardpass() {}
    void backwardpass(const std::shared_ptr<Layer>, double);
    void calcDelta(double);
    
    void flushState();
    void printUnit();
    
    std::vector<double>* getWeights() { return &weights; }
    std::vector<double>* getOutput() const;
    double getDelta() { return delta; }
    double getOutputWeightToCellInPrevLayer(long);
private:
    double alpha;
    std::vector<double> weights;
    
    double net;
    double output;
    
    double delta;
};


#endif /* neuron_h */
