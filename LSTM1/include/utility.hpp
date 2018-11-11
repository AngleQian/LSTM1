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
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <iterator>
#include <algorithm>
#include <math.h>
#include <random>

namespace utility {
    double f(double);
    double df(double);
    double g(double);
    double dg(double);
    double h(double);
    double dh(double);
    double nofunc(double x);
    double dnofunc(double x);
    
    template<typename T>
    void printVector(const T& t){
        std::copy(t.cbegin(), t.cend(), std::ostream_iterator<typename T::value_type>(std::cout, " "));
    }
    
    template<typename T>
    void printVectorInVector(const T& t){
        std::for_each(t.cbegin(), t.cend(), printVector<typename T::value_type>);
    }
    
    template<typename T>
    std::string vectorToString(const std::vector<T>& v){
        std::ostringstream oss;
        std::copy(v.begin(), v.end()-1, std::ostream_iterator<T>(oss, ","));
        oss << v.back();
        return oss.str();
    }

    double getRandomWeight(double, double);
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
    double radius = 1;
    double center = 0;
    
    double translationCoeff;
    double squashCoeff;
};

class NoTransform : public Transform {
public:
    double transformFromPrice(double x) { return x; }
    double transformToPrice(double x) { return x; }
};

#endif /* utility_hpp */
