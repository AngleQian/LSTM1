//
//  layer.cpp
//  LSTM1
//
//  Created by Angle Qian on 11/09/2018.
//  Copyright © 2018 Angle Qian. All rights reserved.
//

#include "../include/layer.hpp"

Layer::Layer(){
    units = std::vector<std::shared_ptr<Unit>>();
}

// layer constructor for regular layers
Layer::Layer(int neuronsPerLayer, double alpha){
    Layer();
    for(int i = 0; i != neuronsPerLayer; ++i){
        std::shared_ptr<Neuron> neuron(new Neuron(alpha));
        std::shared_ptr<Unit> unit(neuron);
        units.push_back(unit);
    }
}


// layer constructor for LSTM layers
Layer::Layer(int blocksPerLayer, int cellsPerBlock, double alpha, long noOfSourceUnits) {
    Layer();
    for(int i = 0; i != blocksPerLayer; ++i){
        std::shared_ptr<MemoryBlock> memoryBlock(new MemoryBlock(cellsPerBlock, alpha, noOfSourceUnits));
        std::shared_ptr<Unit> unit(memoryBlock);
        units.push_back(unit);
    }
}

// forwardpass for input layer, receives input vector
void Layer::forwardpass(const std::vector<double>& input){
    for(int i = 0; i != units.size(); ++i){
        std::shared_ptr<Unit> unit = units[i];
        std::shared_ptr<Neuron> neuron = std::dynamic_pointer_cast<Neuron>(unit);
        // each input neuron recieves 1 input
        neuron->forwardpass(input[i]);
    }
}

// forwardpass for other layers, receives ptr to the previous layer
void Layer::forwardpass(const std::shared_ptr<Layer> prevLayer){
    std::vector<double>* inputs = prevLayer->getOutput();
    for(std::shared_ptr<Unit> unit : units){
        unit->forwardpass(*inputs);
    }
}

// backward pass for output layer, receives vector of external errors e_k(t)
void Layer::backwardpass(const std::shared_ptr<Layer> prevLayer, const std::vector<double>& externalError){
    for(int i = 0; i != units.size(); ++i){
        std::shared_ptr<Unit> unit = units[i];
        std::shared_ptr<Neuron> neuron = std::dynamic_pointer_cast<Neuron>(unit);
        neuron->backwardpass(prevLayer, externalError[i]);
    }
}

// backward pass for hidden layers
void Layer::backwardpass(const std::shared_ptr<Layer> prevLayer, const std::shared_ptr<Layer> nextLayer){
    for(int i = 0; i != units.size(); ++i){
        std::shared_ptr<Unit> unit = units[i];
        std::shared_ptr<MemoryBlock> memoryBlock = std::dynamic_pointer_cast<MemoryBlock>(unit);
        memoryBlock->backwardpass(prevLayer, nextLayer, i);
    }
}

// gets the output of the layer
std::vector<double>* Layer::getOutput() const{
    std::vector<double>* output = new std::vector<double>();
    for(std::shared_ptr<Unit> unit : units){
        std::vector<double>* unitOutput = unit->getOutput();
        output->insert(output->end(), unitOutput->begin(), unitOutput->end());
    }
    return output;
}

void Layer::printLayer(){
    for(int i = 0; i != units.size(); i++){
        std::cout << "Unit: " << i+1 << std::endl;
        units[i]->printUnit();
        std::cout << std::endl;
    }
}
