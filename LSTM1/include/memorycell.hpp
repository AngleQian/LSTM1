//
//  memorycell.hpp
//  LSTM1
//
//  Created by Angle Qian on 12/09/2018.
//  Copyright © 2018 Angle Qian. All rights reserved.
//

#ifndef memorycell_hpp
#define memorycell_hpp

#include <stdio.h>
#include <vector>
#include <memory>
#include <stdlib.h>

#include "utility.hpp"

class Layer;
class MemoryBlock;

class MemoryCell {
public:
    MemoryCell(double, long);
    
    void forwardpass(const std::vector<double>&, double, double, double);
    
    void backwardpass(const std::shared_ptr<Layer>, MemoryBlock*, long);
    
    void calcDelta(const std::shared_ptr<Layer>, long);
    void calcInternalErrorGatePartials(double*, double*, const std::shared_ptr<Layer>, const std::shared_ptr<Layer>, MemoryBlock*, long, int);
    void calcInternalError(const std::shared_ptr<Layer>, MemoryBlock*, long);
    
    void printCell();
    
    std::vector<double>* getCellStateWeights() { return &cellStateWeights; }
    double getOutput() { return output; }
    double getDelta() { return delta; }
private:
    double alpha;
    std::vector<double> cellStateWeights;
    
    double cellStateNet;
    double cellStateCandidate;
    double cellStateCurrent;
    double cellStatePast;
    
    double output;
    
    double delta;
    double internalError;
    
    std::vector<double> internalErrorInputPartialPast;
    std::vector<double> internalErrorForgetPartialPast;
    std::vector<double> internalErrorCellStatePartialPast;
};

#endif /* memorycell_hpp */
