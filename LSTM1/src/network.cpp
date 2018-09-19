//
//  network.cpp
//  LSTM1
//
//  Created by Angle Qian on 11/09/2018.
//  Copyright © 2018 Angle Qian. All rights reserved.
//

#include "../include/network.hpp"

Network::Network(Transform* transform, std::vector<int> topology, int cellsPerBlock,
                 std::vector< std::vector<double> > trainingInputs,
                 std::vector< std::vector<double> > validationInputs, std::vector< std::vector<double> > trainingOutputs, std::vector< std::vector<double> > validationOutputs): transform(transform), trainingInputs(trainingInputs), validationInputs(validationInputs), trainingOutputs(trainingOutputs), validationOutputs(validationOutputs) {
    
    std::cout << "Creating network" << std::endl;
    
    if(topology[0] != trainingInputs[0].size() ||
       topology[topology.size() - 1] != trainingOutputs[0].size() ||
       topology[0] != validationInputs[0].size() ||
       topology[topology.size() - 1] != validationOutputs[0].size()){
        std::cout << "topolgy doesn't match data" << std::endl;
        abort();
    }
    
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
    
    weightInit(cellsPerBlock);
}

void Network::weightInit(int cellsPerBlock){
    std::cout << "Initalizing Weights" << std::endl;
    
    // input neurons directly recieve input vector, so with weight 1
    std::vector<std::shared_ptr<Unit>>* inputLayerUnits = layers[0]->getUnits();
    for(int i = 0; i != inputLayerUnits->size(); ++i){
        std::shared_ptr<Unit> unit = inputLayerUnits->at(i);
        std::shared_ptr<Neuron> neuron = std::dynamic_pointer_cast<Neuron>(unit);
        neuron->getWeights()->push_back(1);
    }
    
    long noOfSourceUnits = layers[0]->getUnits()->size() * cellsPerBlock;

    for(int i = 1; i != layers.size() - 1; ++i){
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

        noOfSourceUnits = hiddenLayerUnits->size() * cellsPerBlock;
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

void Network::train(){
    for(int i = 0; i != trainingInputs.size(); ++i){
        std::cout << "Cycle " << i+1 << std::endl;
        utility::printVector(trainingInputs[i]);
        std::cout << std::endl;
        std::vector<double> output = forwardpass(trainingInputs[i]);
        utility::printVector(output);
        std::cout << std::endl;
        std::cout << std::endl;
    }
    
}

std::vector<double> Network::forwardpass(const std::vector<double>& inputRaw){
    std::vector<double> input = std::vector<double>();
    input.push_back(transform->transformFromPrice(inputRaw[0]));

    std::shared_ptr<Layer> inputLayer = layers[0];
    inputLayer->forwardpass(input);
    
    std::shared_ptr<Layer> prevLayer = inputLayer;
    
    for(int i = 1; i != layers.size() - 1; ++i){
        std::shared_ptr<Layer> hiddenLayer = layers[i];
        hiddenLayer->forwardpass(prevLayer);
        prevLayer = hiddenLayer;
    }
    
    std::shared_ptr<Layer> outputLayer = layers[layers.size() - 1];
    outputLayer->forwardpass(prevLayer);
    
    std::vector<double> rawOutputs = outputLayer->getOutput();
    std::vector<double> outputs = std::vector<double>();
    for(double rawOutput : rawOutputs){
        outputs.push_back(transform->transformToPrice(rawOutput));
    }
    
    return outputs;
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
