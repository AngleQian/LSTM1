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
#include "utility.hpp"

class Neuron : public Unit {
public:
    Neuron();
    
    void forwardpass(double);
    void forwardpass(const std::vector<double>&);
    void printUnit();
    
    std::vector<double>* getWeights() { return &weights; }
    std::vector<double>* getOutput() const;
private:
    std::vector<double> weights;
    double output;
    
};


#endif /* neuron_h */
