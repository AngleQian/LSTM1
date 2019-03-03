//
//  neuron.cpp
//  LSTM1
//
//  Created by Angle Qian on 15/09/2018.
//  Copyright © 2018 Angle Qian. All rights reserved.
//

#include "../include/neuron.hpp"
#include "../include/layer.hpp"

Neuron::Neuron(double alpha) : alpha(alpha) {
    weights = std::make_shared<std::vector<double>>();
}

void Neuron::forwardpass(double input){
    std::vector<double> inputs = std::vector<double>();
    inputs.push_back(input);
    forwardpass(inputs);
}

void Neuron::forwardpass(const std::vector<double>& inputs){
    net = 0;
    for(int i = 0; i != inputs.size(); ++i){
        net += inputs[i] * weights->at(i);
    }
    output = utility::nofunc(net);
}

void Neuron::backwardpass(const std::shared_ptr<Layer> prevLayer, double externalError){
    calcDelta(externalError);
    
    for(int i = 0; i != weights->size(); ++i){
        double deltaWeight = alpha * delta * prevLayer->getOutput()->at(i);
        weights->at(i) += utility::clipping(deltaWeight);
    }
}

void Neuron::calcDelta(double externalError){
    delta = utility::dnofunc(net) * externalError;
}

void Neuron::flushState(){

}

void Neuron::printUnit(){
    std::cout << "W: ";
    utility::printVector(*weights);
    std::cout << "; net: " << net;
    std::cout << "; y: " << output;
    std::cout << "; delta: " << delta;
}

std::shared_ptr<std::vector<double>> Neuron::getOutput() const{
    std::shared_ptr<std::vector<double>> outputVec(new std::vector<double>());
    outputVec->push_back(output);
    return outputVec;
}

double Neuron::getOutputWeightToCellInPrevLayer(long cellPosition){
    return weights->at(cellPosition);
}
