//
//  utility.hpp
//  LSTM1
//
//  Created by Angle Qian on 12/09/2018.
//  Copyright © 2018 Angle Qian. All rights reserved.
//

#ifndef utility_hpp
#define utility_hpp

#include <stdio.h>
#include <iostream>
#include <vector>

#include <math.h>

namespace utility {
    double f(double);
    double g(double);
    double h(double);
    
    void printVector(const std::vector<double> &);
}

class Transform {
public:
    virtual double transformFromPrice(double) = 0;
    virtual double transformToPrice(double) = 0;
};

class TransformLinear : public Transform {
public:
    TransformLinear(double, double, double);
    double transformFromPrice(double);
    double transformToPrice(double);
private:
    const double radius = 3;
    const double center = 0;
    
    double translationCoeff;
    double squashCoeff;
};

#endif /* utility_hpp */
