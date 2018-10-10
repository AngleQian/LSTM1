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
    cellStateWeights = std::vector<double>();
    cellStatePast = 0;
    
    internalErrorInputPartialPast = std::vector<double>();
    internalErrorForgetPartialPast = std::vector<double>();
    internalErrorCellStatePartialPast = std::vector<double>();
    
    for(int i = 0; i != noOfSourceUnits; ++i){
        internalErrorInputPartialPast.push_back(0);
        internalErrorForgetPartialPast.push_back(0);
        internalErrorCellStatePartialPast.push_back(0);
    }
}

void MemoryCell::forwardpass(const std::vector<double> & inputs, double inputGate, double forgetGate, double outputGate){
    cellStateNet = 0;
    for(int i = 0; i != inputs.size(); ++i){
        cellStateNet += inputs[i] * cellStateWeights[i];
    }
    cellStateCandidate = utility::g(cellStateNet);

    cellStateCurrent = forgetGate * cellStatePast + inputGate * cellStateCandidate;
    cellStatePast = cellStateCurrent;
    
    output = outputGate * utility::h(cellStateCurrent);
}

void MemoryCell::backwardpass(const std::shared_ptr<Layer> prevLayer, MemoryBlock* memoryBlock, long cellPosition){
    double deltaCellStateWeight;
    double internalErrorCellStatePartial;

    for(int i = 0; i != cellStateWeights.size(); ++i){
        internalErrorCellStatePartial = internalError * (internalErrorCellStatePartialPast[i] * memoryBlock->getForgetGate() + utility::dg(cellStateNet) * memoryBlock->getInputGate() * prevLayer->getOutput()->at(i));
        deltaCellStateWeight = alpha * internalErrorCellStatePartial;
        cellStateWeights[i] += deltaCellStateWeight;
    }
}

void MemoryCell::calcDelta(const std::shared_ptr<Layer> nextLayer, long cellPosition){
    double factor = utility::h(cellStateCurrent);
    delta = 0;
    for(std::shared_ptr<Unit> unit : *nextLayer->getUnits()){
        delta += factor * unit->getDelta() * unit->getOutputWeightToCellInPrevLayer(cellPosition);
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
    for(std::shared_ptr<Unit> unit : *nextLayer->getUnits()){
        internalError += factor * (unit->getDelta() * unit->getOutputWeightToCellInPrevLayer(cellPosition));
    }
}

void MemoryCell::printCell(){
    std::cout << "Wcell: ";
    utility::printVector(cellStateWeights);
}
