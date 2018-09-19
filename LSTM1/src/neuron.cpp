//
//  neuron.cpp
//  LSTM1
//
//  Created by Angle Qian on 15/09/2018.
//  Copyright © 2018 Angle Qian. All rights reserved.
//

#include "../include/neuron.hpp"

Neuron::Neuron(){
    
}

void Neuron::forwardpass(double input){
    std::vector<double> inputs = std::vector<double>();
    inputs.push_back(input);
    forwardpass(inputs);
}

void Neuron::forwardpass(const std::vector<double>& inputs){
    double net = 0;
    for(int i = 0; i != inputs.size(); ++i){
        net += inputs[i] * weights[i];
    }
    output = utility::f(net);
}

std::vector<double>* Neuron::getOutput() const{
    std::vector<double>* outputVec = new std::vector<double>();
    outputVec->push_back(output);
    return outputVec;
}

void Neuron::printUnit(){
    std::cout << "W: ";
    utility::printVector(weights);
}
