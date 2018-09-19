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

class MemoryCell {
public:
    MemoryCell();
    
    void forwardpass(const std::vector<double>&, double, double, double);
    
    void printCell();
    
    std::vector<double>* getCellStateWeights() { return &cellStateWeights; }
    double getOutput() { return output; }
private:
    double cellStateCurrent;
    double cellStatePast;
    std::vector<double> cellStateWeights;
    double output;
};

#endif /* memorycell_hpp */
