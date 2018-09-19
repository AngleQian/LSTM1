//
//  unit.hpp
//  LSTM1
//
//  Created by Angle Qian on 15/09/2018.
//  Copyright © 2018 Angle Qian. All rights reserved.
//

#ifndef unit_h
#define unit_h

#include <stdio.h>
#include <vector>
#include <stdlib.h>

// Base class
class Unit {
public:
    virtual ~Unit(){};
    virtual void forwardpass(const std::vector<double>&) = 0;
    virtual void printUnit() = 0;
    virtual std::vector<double>* getOutput() const = 0;
};


#endif /* unit_h */
