//
//  network.cpp
//  LSTM1
//
//  Created by Angle Qian on 11/09/2018.
//  Copyright © 2018 Angle Qian. All rights reserved.
//

#include "../include/network.hpp"

Network::Network(Transform* transform, std::vector<int> topology, int cellsPerBlock, double alpha,
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
    std::shared_ptr<Layer> inputLayer(new Layer(topology[0], alpha));
    Network::layers.push_back(inputLayer);
    
    std::shared_ptr<Layer> prevLayer = inputLayer;
    
    //create first LSTM layer
    std::shared_ptr<Layer> layer(new Layer(topology[1], cellsPerBlock, alpha, prevLayer->getUnits()->size()));
    prevLayer = layer;
    
    // create subsequent LSTM layers
    for(int i = 2; i != topology.size() - 1; ++i){
        std::shared_ptr<Layer> layer(new Layer(topology[i], cellsPerBlock, alpha, prevLayer->getUnits()->size() * cellsPerBlock));
        Network::layers.push_back(layer);
        prevLayer = layer;
    }
    
    // create output layer
    std::shared_ptr<Layer> outputLayer(new Layer(topology[topology.size() - 1], alpha));
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
    
    long noOfSourceUnits = layers[0]->getUnits()->size();

    for(int i = 1; i != layers.size() - 1; ++i){
        std::vector<std::shared_ptr<Unit>>* hiddenLayerUnits = layers[i]->getUnits();
        
        for(int j = 0; j != hiddenLayerUnits->size(); ++j){
            std::shared_ptr<Unit> unit = hiddenLayerUnits->at(j);
            std::shared_ptr<MemoryBlock> memoryBlock = std::dynamic_pointer_cast<MemoryBlock>(unit);
            
            for(int k = 0; k != noOfSourceUnits; ++k){
                memoryBlock->getInputGateWeights()->push_back(utility::getRandomWeight(0, 5));
                memoryBlock->getOutputGateWeights()->push_back(utility::getRandomWeight(0, 5));
                memoryBlock->getForgetGateWeights()->push_back(utility::getRandomWeight(-5, 0));
                
                for(std::shared_ptr<MemoryCell> memoryCell : *(memoryBlock->getMemoryCells())){
                    memoryCell->getCellStateWeights()->push_back(utility::getRandomWeight(0, 5));
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
            neuron->getWeights()->push_back(utility::getRandomWeight(-0.2, 0.2));
        }
    }
}

void Network::train(){
    std::ofstream file1;
    std::string directory;
    directory = dataprocessing::baseDirectory + "output/" + "trainingOutput.txt";
    file1.open(directory);
    
    std::cout << "Cycle " << 1 << std::endl;
    
    std::vector<double> outputs = forwardpass(trainingInputs[0]);
    
    utility::printVector(trainingInputs[0]);
    std::cout << std::endl;
    utility::printVector(outputs);
    std::cout << std::endl;
    std::cout << std::endl;
    
    file1 << trainingInputs[0][0] << "," << outputs[0] << "\n";
    
    for(int i = 1; i != trainingInputs.size(); ++i){
        std::cout << "Cycle" << i+1 << ": " << std::endl;
        
        outputs = forwardpass(trainingInputs[i]);
        
        std::vector<double> targetOutputs = trainingOutputs[i-1];
        backwardpass(targetOutputs, outputs);
        
        utility::printVector(trainingInputs[i]);
        std::cout << std::endl;
        utility::printVector(outputs);
        std::cout << std::endl;
        std::cout << std::endl;
        
        file1 << trainingInputs[i][0] << "," << outputs[0] << "\n";
        
    }
    file1.close();
    
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
    
    std::vector<double>* rawOutputs = outputLayer->getOutput();
    
    std::vector<double> outputs = std::vector<double>();
    for(double rawOutput : *rawOutputs){
        outputs.push_back(transform->transformToPrice(rawOutput));
    }
    
    return outputs;
}

void Network::backwardpass(const std::vector<double>& targetOutput, const std::vector<double>& output){
    std::vector<double> externalError = std::vector<double>();
    for(int i = 0; i != targetOutput.size(); ++i){
        externalError.push_back(transform->transformFromPrice(targetOutput[i]) - transform->transformFromPrice(output[i]));
    }
    
    std::cout << "Error: ";
    utility::printVector(externalError);
    std::cout << std::endl;
    
    std::shared_ptr<Layer> outputLayer = layers[layers.size()-1];
    std::shared_ptr<Layer> prevLayer = layers[layers.size()-2];
    outputLayer->backwardpass(prevLayer, externalError);
    
    std::shared_ptr<Layer> nextLayer;
    std::shared_ptr<Layer> hiddenLayer;
    
    for(long i = layers.size() - 2; i != 0; --i){
        nextLayer = layers[i+1];
        hiddenLayer = layers[i];
        prevLayer = layers[i-1];
        hiddenLayer->backwardpass(prevLayer, nextLayer);
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
