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

void Neuron::printUnit(){
    std::cout << "W: ";
    utility::printVector(weights);
}
