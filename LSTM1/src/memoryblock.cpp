//
//  neuron.cpp
//  LSTM1
//
//  Created by Angle Qian on 11/09/2018.
//  Copyright © 2018 Angle Qian. All rights reserved.
//

#include "../include/memoryblock.hpp"
#include "../include/layer.hpp"

MemoryBlock::MemoryBlock(int cellsPerBlock, double alpha, long noOfSourceUnits) : alpha(alpha) {
    for(int i = 0; i != cellsPerBlock; ++i){
        std::shared_ptr<MemoryCell> memoryCell = std::make_shared<MemoryCell>(alpha, noOfSourceUnits);
        memoryCells.push_back(memoryCell);
    }
    inputGateWeights = std::make_shared<std::vector<double>>();
    forgetGateWeights = std::make_shared<std::vector<double>>();
    outputGateWeights = std::make_shared<std::vector<double>>();
    pendingInputGateWeights = std::make_shared<std::vector<double>>();
    pendingForgetGateWeights = std::make_shared<std::vector<double>>();
    pendingOutputGateWeights = std::make_shared<std::vector<double>>();
    
    for(int i = 0; i != noOfSourceUnits; ++i) {
        pendingInputGateWeights->push_back(0);
        pendingForgetGateWeights->push_back(0);
        pendingOutputGateWeights->push_back(0);
    }
}

void MemoryBlock::forwardpass(const std::vector<double> & inputs){
    inputNet = 0;
    forgetNet = 0;
    outputNet = 0;
    
    for(int i = 0; i != inputs.size(); ++i){
        inputNet += inputs[i] * inputGateWeights->at(i);
        forgetNet += inputs[i] * forgetGateWeights->at(i);
        outputNet += inputs[i] * outputGateWeights->at(i);
    }
    
    inputGate = utility::f(inputNet);
    forgetGate = utility::f(forgetNet);
    outputGate = utility::f(outputNet);
    
    for(std::shared_ptr<MemoryCell> memoryCell : memoryCells){
        memoryCell->forwardpass(inputs, inputGate, forgetGate, outputGate);
    }
}

void MemoryBlock::backwardpass(const std::shared_ptr<Layer> prevLayer, const std::shared_ptr<Layer> nextLayer, int blockPosition){
    calcDelta(nextLayer, blockPosition);
    
    std::ofstream file;
    std::string directory = "/Users/angleqian/Drive Sync/Extended Essay/LSTM1/LSTM1/output/debug1.txt";
    
    file.open(directory, std::ios_base::app);
    
    double deltaOutputWeight;
    for(int i = 0; i != outputGateWeights->size(); ++i){
        deltaOutputWeight = alpha * delta * prevLayer->getOutput()->at(i);
//      std::cout << "dOW: " << deltaOutputWeight << std::endl;
        pendingOutputGateWeights->at(i) = outputGateWeights->at(i) + utility::clipping(deltaOutputWeight);
//       file << deltaOutputWeight << "\n";
    }
    
    double deltaInputWeight;
    double deltaForgetWeight;
    
    double internalErrorInputPartial;
    double internalErrorForgetPartial;
    
    for(int i = 0; i != inputGateWeights->size(); ++i){
        calcInternalErrorGatePartials(&internalErrorInputPartial, &internalErrorForgetPartial, prevLayer, nextLayer, this, blockPosition, i);
        deltaInputWeight = alpha * internalErrorInputPartial;
//        std::cout << "dIW: " << deltaInputWeight << std::endl;
        pendingInputGateWeights->at(i) = inputGateWeights->at(i) + utility::clipping(deltaInputWeight);
        deltaForgetWeight = alpha * internalErrorForgetPartial;
//        std::cout << "dFW: " << deltaForgetWeight << std::endl;
        pendingForgetGateWeights->at(i) = forgetGateWeights->at(i) + utility::clipping(deltaForgetWeight);
    }
    
    long cellPosition;

    for(int i = 0; i != memoryCells.size(); i++){
        cellPosition = blockPosition * memoryCells.size() + i;
        memoryCells[i]->backwardpass(prevLayer, this, cellPosition);
    }
}

void MemoryBlock::calcDelta(const std::shared_ptr<Layer> nextLayer, int blockPosition){
    delta = 0;
    long cellPosition;
    for(int i = 0; i != memoryCells.size(); ++i){
        cellPosition = blockPosition * memoryCells.size() + i;
        memoryCells[i]->calcDelta(nextLayer, cellPosition);
        delta += memoryCells[i]->getDelta();
    }
    delta *= utility::df(outputNet);
}

void MemoryBlock::calcInternalErrorGatePartials(double* internalErrorInputPartial, double* internalErrorForgetPartial, const std::shared_ptr<Layer> prevLayer, const std::shared_ptr<Layer> nextLayer, MemoryBlock * memoryBlock, int blockPosition, int sourceUnitIndex){
    *internalErrorInputPartial = 0;
    *internalErrorForgetPartial = 0;
    
    double iEIPtemp;
    double iEFPtemp;
    
    long cellPosition;
    
    for(int i = 0; i != memoryCells.size(); ++i){
        cellPosition = blockPosition * memoryCells.size() + i;
        memoryCells[i]->calcInternalErrorGatePartials(&iEIPtemp, &iEFPtemp, prevLayer, nextLayer, memoryBlock, cellPosition, sourceUnitIndex);
        *internalErrorInputPartial += iEIPtemp;
        *internalErrorForgetPartial += iEFPtemp;
    }
}

void MemoryBlock::applyWeightChanges() {
    for(int i = 0; i != inputGateWeights->size(); ++i) {
        inputGateWeights->at(i) = pendingInputGateWeights->at(i);
        forgetGateWeights->at(i) = pendingForgetGateWeights->at(i);
        outputGateWeights->at(i) = pendingOutputGateWeights->at(i);
    }
    
    for(std::shared_ptr<MemoryCell> memoryCell : memoryCells) {
        memoryCell->applyWeightChanges();
    }
}

void MemoryBlock::flushState(){
    for(int i = 0; i != memoryCells.size(); ++i){
        memoryCells[i]->flushState();
    }
}

void MemoryBlock::printUnit(){
    std::cout << "Wi: ";
    utility::printVector(*inputGateWeights);
    std::cout << "; Wf: ";
    utility::printVector(*forgetGateWeights);
    std::cout << "; Wo: ";
    utility::printVector(*outputGateWeights);
    std::cout << "; Yi: " << inputGate;
    std::cout << "; Yf: " << forgetGate;
    std::cout << "; Yo: " << outputGate;
    

    std::cout << std::endl;
    
    std::cout << "Delta: " << delta;
    std::cout << std::endl;
    
    for(int i = 0; i != memoryCells.size(); ++i){
        std::cout << "Cell: " << i+1 << std::endl;
        memoryCells[i]->printCell();
        std::cout << std::endl;
    }
}

std::shared_ptr<std::vector<double>> MemoryBlock::getOutput() const{
    std::shared_ptr<std::vector<double>> output = std::make_shared<std::vector<double>>();
    for(std::shared_ptr<MemoryCell> memoryCell : memoryCells){
        output->push_back(memoryCell->getOutput());
    }
    return output;
}

double MemoryBlock::getOutputWeightToCellInPrevLayer(long cellPosition){
    return outputGateWeights->at(cellPosition);
}
