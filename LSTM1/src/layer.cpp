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
Layer::Layer(int neuronsPerLayer){
    Layer();
    for(int i = 0; i != neuronsPerLayer; ++i){
        std::shared_ptr<Neuron> neuron(new Neuron());
        std::shared_ptr<Unit> unit(neuron);
        units.push_back(unit);
    }
}


// layer constructor for LSTM layers
Layer::Layer(int blocksPerLayer, int cellsPerBlock) {
    Layer();
    for(int i = 0; i != blocksPerLayer; ++i){
        std::shared_ptr<MemoryBlock> memoryBlock(new MemoryBlock(cellsPerBlock));
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
    std::vector<double> inputs = prevLayer->getOutput();
    for(std::shared_ptr<Unit> unit : units){
        unit->forwardpass(inputs);
    }
}

// gets the output of the layer, usually used for the output layer
std::vector<double> Layer::getOutput(){
    std::vector<double> output = std::vector<double>();
    for(std::shared_ptr<Unit> unit : units){
        std::vector<double>* unitOutput = unit->getOutput();
        output.insert(output.end(), unitOutput->begin(), unitOutput->end());
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
