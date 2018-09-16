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

// Base class
class Unit {
public:
    virtual ~Unit(){};
    virtual void printUnit() = 0;
};


#endif /* unit_h */
