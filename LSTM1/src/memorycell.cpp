//
//  memorycell.cpp
//  LSTM1
//
//  Created by Angle Qian on 12/09/2018.
//  Copyright © 2018 Angle Qian. All rights reserved.
//

#include "../include/memorycell.hpp"
#include "../include/layer.hpp"
#include "../include/memoryblock.hpp"

MemoryCell::MemoryCell(double alpha, long noOfSourceUnits) : alpha(alpha) {
    cellStateWeights = std::make_shared<std::vector<double>>();
    pendingCellStateWeights = std::make_shared<std::vector<double>>();
    cellStatePast = 0;
    
    internalErrorInputPartialPast = std::vector<double>();
    internalErrorForgetPartialPast = std::vector<double>();
    internalErrorCellStatePartialPast = std::vector<double>();
    
    for(int i = 0; i != noOfSourceUnits; ++i){
        internalErrorInputPartialPast.push_back(0);
        internalErrorForgetPartialPast.push_back(0);
        internalErrorCellStatePartialPast.push_back(0);
        
        pendingCellStateWeights->push_back(0);
    }
    
}

void MemoryCell::forwardpass(const std::vector<double> & inputs, double inputGate, double forgetGate, double outputGate){
    cellStateNet = 0;
    for(int i = 0; i != inputs.size(); ++i){
        cellStateNet += inputs[i] * cellStateWeights->at(i);
    }
    cellStateCandidate = utility::g(cellStateNet);

    cellStateCurrent = forgetGate * cellStatePast + inputGate * cellStateCandidate;
    
    output = outputGate * utility::h(cellStateCurrent);
}

void MemoryCell::backwardpass(const std::shared_ptr<Layer> prevLayer, MemoryBlock* memoryBlock, long cellPosition){
    double deltaCellStateWeight;
    double internalErrorCellStatePartial;

    for(int i = 0; i != cellStateWeights->size(); ++i){
        internalErrorCellStatePartial = internalError * (internalErrorCellStatePartialPast[i] * memoryBlock->getForgetGate() + utility::dg(cellStateNet) * memoryBlock->getInputGate() * prevLayer->getOutput()->at(i));
        
        internalErrorCellStatePartialPast[i] = internalErrorCellStatePartial;
        
        deltaCellStateWeight = alpha * internalErrorCellStatePartial;
        pendingCellStateWeights->at(i) = cellStateWeights->at(i) + utility::clipping(deltaCellStateWeight);
    }
}

void MemoryCell::calcDelta(const std::shared_ptr<Layer> nextLayer, long cellPosition){
    delta = 0;
    std::shared_ptr< std::vector< std::shared_ptr<Unit> > > units = nextLayer->getUnits();
    for(int i = 0; i != units->size(); ++i){
        std::shared_ptr<Unit> unit = units->at(i);
        delta += utility::h(cellStateCurrent) * unit->getDelta() * unit->getOutputWeightToCellInPrevLayer(cellPosition);
    }
}

void MemoryCell::calcInternalErrorGatePartials(double* internalErrorInputPartial, double* internalErrorForgetPartial, const std::shared_ptr<Layer> prevLayer, const std::shared_ptr<Layer> nextLayer, MemoryBlock* memoryBlock, long cellPosition, int sourceUnitIndex){
    calcInternalError(nextLayer, memoryBlock, cellPosition);
    
    *internalErrorInputPartial = internalError * (internalErrorInputPartialPast[sourceUnitIndex] * memoryBlock->getForgetGate() + cellStateCandidate * utility::df(memoryBlock->getInputNet()) * prevLayer->getOutput()->at(sourceUnitIndex));
    internalErrorInputPartialPast[sourceUnitIndex] = *internalErrorInputPartial;
    
    *internalErrorForgetPartial = internalError * (internalErrorForgetPartialPast[sourceUnitIndex] * memoryBlock->getForgetGate() + cellStatePast * utility::df(memoryBlock->getForgetNet()) * prevLayer->getOutput()->at(sourceUnitIndex));
    internalErrorForgetPartialPast[sourceUnitIndex] = *internalErrorForgetPartial;
}

void MemoryCell::calcInternalError(const std::shared_ptr<Layer> nextLayer, MemoryBlock* memoryBlock, long cellPosition){
    internalError = 0;
    double factor = memoryBlock->getOutputGate() * utility::dh(cellStateCurrent);
    std::vector<std::shared_ptr<Unit>> nextLayerUnits = *nextLayer->getUnits();
    
    for(std::shared_ptr<Unit> unit : nextLayerUnits){
        internalError += factor * unit->getDelta() * unit->getOutputWeightToCellInPrevLayer(cellPosition);
    }
}

void MemoryCell::applyWeightChanges() {
    cellStatePast = cellStateCurrent;
    
    for(int i = 0; i != cellStateWeights->size(); ++i) {
        cellStateWeights->at(i) = pendingCellStateWeights->at(i);
    }
}

void MemoryCell::flushState(){
    cellStatePast = 0;
}

void MemoryCell::printCell(){
    std::cout << "Wcell: ";
    utility::printVector(*cellStateWeights);
    std::cout << "; cellStateNet: " << cellStateNet;
    std::cout << "; cellStateCandidate: " << cellStateCandidate;
    std::cout << "; cellState: " << cellStateCurrent;
    std::cout << "; y: " << output;
    std::cout << "; internalError: " << internalError;
    std:: cout << "; delta: " << delta;
}
