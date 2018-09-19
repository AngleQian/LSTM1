//
//  utility.cpp
//  LSTM1
//
//  Created by Angle Qian on 12/09/2018.
//  Copyright © 2018 Angle Qian. All rights reserved.
//

#include "../include/utility.hpp"

double utility::f(double x) {
    return 1/(1+exp(-x));
}

double utility::g(double x) {
    return 4/(1+exp(-x)) - 2;
}

double utility::h(double x) {
    return 2/(1+exp(-x)) - 1;
}


TransformLinear::TransformLinear(double average, double rangeMin, double rangeMax) {
    double maxDiff = (average-rangeMin) >= (rangeMax-average) ? (average-rangeMin)  : (rangeMax-average);
    squashCoeff = radius/maxDiff;
    translationCoeff = center - squashCoeff * average;
}

double TransformLinear::transformFromPrice(double price) {
    return (price * squashCoeff) + translationCoeff;
}

double TransformLinear::transformToPrice(double x) {
    return (x - translationCoeff) / squashCoeff;
}

