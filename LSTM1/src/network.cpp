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
                 std::vector< std::vector<double> > trainingOutputs, std::vector< std::vector<double> > validationInputs, std::vector< std::vector<double> > validationOutputs): transform(transform), alpha(alpha), trainingInputs(trainingInputs), validationInputs(validationInputs), trainingOutputs(trainingOutputs), validationOutputs(validationOutputs) {
    if(topology[0] != trainingInputs[0].size() ||
       topology[topology.size() - 1] != trainingOutputs[0].size() ||
       topology[0] != validationInputs[0].size() ||
       topology[topology.size() - 1] != validationOutputs[0].size()){
        std::cout << "topology doesn't match data" << std::endl;
        abort();
    }

    // create first input layer
    std::shared_ptr<Layer> inputLayer = std::make_shared<Layer>(topology[0], alpha);
    layers.push_back(inputLayer);

    std::shared_ptr<Layer> prevLayer = inputLayer;

    //create first LSTM layer
    std::shared_ptr<Layer> layer = std::make_shared<Layer>(topology[1], cellsPerBlock, alpha, prevLayer->getUnits()->size());
    layers.push_back(layer);
    prevLayer = layer;

    // create subsequent LSTM layers
    for(int i = 2; i != topology.size() - 1; ++i){
        std::shared_ptr<Layer> layer = std::make_shared<Layer>(topology[i], cellsPerBlock, alpha, prevLayer->getUnits()->size() * cellsPerBlock);
        Network::layers.push_back(layer);
        prevLayer = layer;
    }

    // create output layer
    std::shared_ptr<Layer> outputLayer = std::make_shared<Layer>(topology[topology.size() - 1], alpha);
    Network::layers.push_back(outputLayer);

    weightInit(cellsPerBlock);
}

void Network::weightInit(int cellsPerBlock){
    // initialize weights for input units
    std::shared_ptr<std::vector<std::shared_ptr<Unit>>> inputLayerUnits = layers[0]->getUnits();
    for(int i = 0; i != inputLayerUnits->size(); ++i){
        std::shared_ptr<Unit> unit = inputLayerUnits->at(i);
        std::shared_ptr<Neuron> neuron = std::dynamic_pointer_cast<Neuron>(unit);
        // input neurons directly recieve input vector, so with weight 1
        neuron->getWeights()->push_back(1);
    }

    long noOfSourceUnits = layers[0]->getUnits()->size();

    // initialize weights for hidden units
    for(int i = 1; i != layers.size() - 1; ++i){
        std::shared_ptr<std::vector<std::shared_ptr<Unit>>> hiddenLayerUnits = layers[i]->getUnits();

        for(int j = 0; j != hiddenLayerUnits->size(); ++j){
            std::shared_ptr<Unit> unit = hiddenLayerUnits->at(j);
            std::shared_ptr<MemoryBlock> memoryBlock = std::dynamic_pointer_cast<MemoryBlock>(unit);

            for(int k = 0; k != noOfSourceUnits; ++k){
                memoryBlock->getInputGateWeights()->push_back(utility::getRandomWeight(-5, 0));
                memoryBlock->getOutputGateWeights()->push_back(utility::getRandomWeight(-5, 0));
                memoryBlock->getForgetGateWeights()->push_back(utility::getRandomWeight(0, 5));

                for(std::shared_ptr<MemoryCell> memoryCell : memoryBlock->getMemoryCells()){
                    memoryCell->getCellStateWeights()->push_back(utility::getRandomWeight(-5, 0));
                }
            }
        }

        noOfSourceUnits = hiddenLayerUnits->size() * cellsPerBlock;
    }

    // initialize weights for output units
    std::shared_ptr<std::vector<std::shared_ptr<Unit>>> outputLayerUnits = layers[layers.size() - 1]->getUnits();
    for(int i = 0; i != outputLayerUnits->size(); ++i){
        std::shared_ptr<Unit> unit = outputLayerUnits->at(i);
        std::shared_ptr<Neuron> neuron = std::dynamic_pointer_cast<Neuron>(unit);
        for(int j = 0; j != noOfSourceUnits; j++){
            neuron->getWeights()->push_back(utility::getRandomWeight(-5, 5));
        }
    }
}

void Network::train(){
    std::ofstream file1;
    std::ofstream file2;
    std::string directory = dataprocessing::baseDirectory + "output/" + "trainingOutput.txt";
    std::string directory2 = dataprocessing::baseDirectory + "output/" + "trainingError.txt";
    file1.open(directory);
    file2.open(directory2);

    double error;
    double sumOfError = 0;
    std::vector<double> inputs;
    std::vector<double> targetOutputs;
    std::vector<double> outputs;

    for(int i = 0; i != trainingOutputs.size(); ++i){
        if(i < 3){
            printNetwork();
        }
        inputs = trainingInputs[i];
        outputs = forwardpass(inputs);
        targetOutputs = trainingOutputs[i];
//        error = (double) abs(backwardpass(targetOutputs, outputs)) / targetOutputs[0];
        error = (double) abs(backwardpass(targetOutputs, outputs)) / 1.0;
        sumOfError += error;

//        std::cout << "Training cycle " << i+1 << ": " << std::endl;
//        std::cout << "Input: ";
//        utility::printVector(inputs);
//        std::cout << std::endl << "Target output: ";
//        utility::printVector(targetOutputs);
//        std::cout << std::endl << "Output: ";
//        utility::printVector(outputs);
//        std::cout << std::endl << "Error: " << error;
//        std::cout << std::endl << std::endl;

        file1 << inputs[0] << "," << targetOutputs[0] << "," << outputs[0] << "\n";
        file2 << error << "\n";
    }
//    std::cout << std::endl;
//    std::cout << "Avg error during training: " << (double) sumOfError / (trainingInputs.size()-1) << std::endl;
    file1.close();
    file2.close();
}

double Network::validate(){
    std::ofstream file1;
//    std::ofstream file2;
    std::string directory = dataprocessing::baseDirectory + "output/" + "validationOutput.txt";
//    std::string directory2 = dataprocessing::baseDirectory + "output/" + "validationError.txt";
    file1.open(directory);
//    file2.open(directory2);

//    flushState();

    double error;
    double sumOfError = 0;
    std::vector<double> inputs;
    std::vector<double> targetOutputs;
    std::vector<double> outputs;

    for(int i = 0; i != validationOutputs.size(); ++i){
        inputs = validationInputs[i];
        outputs = forwardpass(inputs);
        targetOutputs = validationOutputs[i];

//        error = (double) abs(targetOutputs[0] - outputs[0]) / targetOutputs[0];
        error = (double) abs(targetOutputs[0] - outputs[0]) / 1.0;
        sumOfError += error;

//        std::cout << "Validation cycle " << i+1 << ": " << std::endl;
//        std::cout << "Input: ";
//        utility::printVector(inputs);
//        std::cout << std::endl << "Target output: ";
//        utility::printVector(targetOutputs);
//        std::cout << std::endl << "Output: ";
//        utility::printVector(outputs);
//        std::cout << std::endl << "Error: " << error;
//        std::cout << std::endl << std::endl;

        file1 << inputs[0] << "," << targetOutputs[0] << "," << outputs[0] << "\n";
//        file2 << error << "\n";
    }
//    std::cout << std::endl;
//    std::cout << "Avg error during validation: " << (double) sumOfError / (validationInputs.size()-1) << std::endl;
    file1.close();
//    file2.close();

    return (double) sumOfError / (validationInputs.size()-1);
}

std::vector<double> Network::forwardpass(const std::vector<double>& inputRaw){
    std::vector<double> input = std::vector<double>();
    for(int i = 0; i != inputRaw.size(); ++i){
        input.push_back(transform->transformFromPrice(inputRaw[i]));
    }

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

    std::shared_ptr<std::vector<double>> rawOutputs = outputLayer->getOutput();

    std::vector<double> outputs = std::vector<double>();
    for(double rawOutput : *rawOutputs){
        outputs.push_back(transform->transformToPrice(rawOutput));
    }

    return outputs;
}

double Network::backwardpass(const std::vector<double>& targetOutput, const std::vector<double>& output){
    std::vector<double> externalError = std::vector<double>();
    for(int i = 0; i != targetOutput.size(); ++i){
       externalError.push_back(transform->transformFromPrice(targetOutput[i]) - transform->transformFromPrice(output[i]));
    }

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

    return externalError[0];
}

void Network::flushState(){
    for(int i = 0; i != layers.size(); ++i){
        layers[i]->flushState();
    }
}

void Network::printNetwork(){
    std::cout << "Printing Network" << std::endl;
    for(int i = 0; i != layers.size(); ++i){
        std::cout << "------------------------------------ Layer: " << i+1;
        std::cout << " ------------------------------------" << std::endl;
        layers[i]->printLayer();
        std::cout << "----------------------------------------------------------------------------------" << std::endl << std::endl << std::endl << std::endl;
    }
}
