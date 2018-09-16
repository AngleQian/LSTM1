//
//  network.cpp
//  LSTM1
//
//  Created by Angle Qian on 11/09/2018.
//  Copyright © 2018 Angle Qian. All rights reserved.
//

#include "../include/network.hpp"

Network::Network(Transform* transform, std::vector<int> topology, int cellsPerBlock,
                 std::vector< std::vector<double> > trainingSet,
                 std::vector< std::vector<double> > validationSet): transform(transform), trainingSet(trainingSet), validationSet(validationSet){
    std::cout << "Creating network" << std::endl;
    
    // create first input layer    
    std::shared_ptr<Layer> inputLayer(new Layer(topology[0]));
    Network::layers.push_back(inputLayer);
    
    // create hidden LSTM layers
    for(int i = 1; i != topology.size() - 1; ++i){
        std::shared_ptr<Layer> layer(new Layer(topology[i], cellsPerBlock));
        Network::layers.push_back(layer);
    }
    
    // create output layer
    std::shared_ptr<Layer> outputLayer(new Layer(topology[topology.size() - 1]));
    Network::layers.push_back(outputLayer);
    
    weightInit();
}

void Network::weightInit(){
    std::cout << "Initalizing Weights" << std::endl;
    
    // input neurons directly recieve input vector, so with weight 1
    std::vector<std::shared_ptr<Unit>>* inputLayerUnits = layers[0]->getUnits();
    for(int i = 0; i != inputLayerUnits->size(); ++i){
        std::shared_ptr<Unit> unit = inputLayerUnits->at(i);
        std::shared_ptr<Neuron> neuron = std::dynamic_pointer_cast<Neuron>(unit);
        neuron->getWeights()->push_back(1);
    }
    
    long noOfSourceUnits = layers[0]->getUnits()->size();

    for(int i = 1; i != layers.size() - 1; i++){
        std::vector<std::shared_ptr<Unit>>* hiddenLayerUnits = layers[i]->getUnits();
        
        for(int j = 0; j != hiddenLayerUnits->size(); ++j){
            std::shared_ptr<Unit> unit = hiddenLayerUnits->at(j);
            std::shared_ptr<MemoryBlock> memoryBlock = std::dynamic_pointer_cast<MemoryBlock>(unit);
            
            for(int k = 0; k != noOfSourceUnits; ++k){
                memoryBlock->getInputGateWeights()->push_back(-1);
                memoryBlock->getOutputGateWeights()->push_back(-1);
                memoryBlock->getForgetGateWeights()->push_back(1);
                
                for(std::shared_ptr<MemoryCell> memoryCell : *(memoryBlock->getMemoryCells())){
                    memoryCell->getCellStateWeights()->push_back(1);
                }
            }
        }

        noOfSourceUnits = hiddenLayerUnits->size();
    }

    std::vector<std::shared_ptr<Unit>>* outputLayerUnits = layers[layers.size() - 1]->getUnits();
    for(int i = 0; i != outputLayerUnits->size(); ++i){
        std::shared_ptr<Unit> unit = outputLayerUnits->at(i);
        std::shared_ptr<Neuron> neuron = std::dynamic_pointer_cast<Neuron>(unit);
        for(int j = 0; j != noOfSourceUnits; j++){
            neuron->getWeights()->push_back(1);
        }
    }
}

void Network::printNetwork(){
    std::cout << "Printing Network" << std::endl;
    for(int i = 0; i != layers.size(); i++){
        std::cout << "------------------------------------ Layer: " << i+1;
        std::cout << " ------------------------------------" << std::endl;
        layers[i]->printLayer();
        std::cout << "----------------------------------------------------------------------------------" << std::endl << std::endl << std::endl << std::endl;
    }
}
