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

Layer::Layer(int neuronsPerLayer){
    Layer();
    for(int i = 0; i != neuronsPerLayer; ++i){
        std::shared_ptr<Neuron> neuron(new Neuron());
        std::shared_ptr<Unit> unit(move(neuron));
        units.push_back(unit);
    }
}

Layer::Layer(int blocksPerLayer, int cellsPerBlock) {
    Layer();
    for(int i = 0; i != blocksPerLayer; ++i){
        std::shared_ptr<MemoryBlock> memoryBlock(new MemoryBlock(cellsPerBlock));
        std::shared_ptr<Unit> unit(memoryBlock);
        units.push_back(unit);
    }
}

void Layer::printLayer(){
    for(int i = 0; i != units.size(); i++){
        std::cout << "Unit: " << i+1 << std::endl;
        units[i]->printUnit();
        std::cout << std::endl;
    }
}
