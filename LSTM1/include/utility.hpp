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
#include <iterator>
#include <algorithm>
#include <math.h>

namespace utility {
    double f(double);
    double g(double);
    double h(double);
    
    template<typename T>
    void printVector(const T& t) {
        std::copy(t.cbegin(), t.cend(), std::ostream_iterator<typename T::value_type>(std::cout, " "));
    }
    
    template<typename T>
    void printVectorInVector(const T& t) {
        std::for_each(t.cbegin(), t.cend(), printVector<typename T::value_type>);
    }
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
