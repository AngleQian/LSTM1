//
//  utility.cpp
//  LSTM1
//
//  Created by Angle Qian on 12/09/2018.
//  Copyright © 2018 Angle Qian. All rights reserved.
//

#include "../include/utility.hpp"

double utility::f(double x){
    return 1/(1+exp(-x));
}

double utility::df(double x){
    return f(x)*(1-f(x));
}

double utility::g(double x){
    return 4/(1+exp(-x)) - 2;
}

double utility::dg(double x){
    return 4 * df(x);
}

double utility::h(double x){
    return 2/(1+exp(-x)) - 1;
}

double utility::dh(double x){
    return 2 * df(x);
}

double utility::getRandomWeight(double a, double b){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> weightDist(a, b);
    return weightDist(gen);
}

TransformLinear::TransformLinear(double average, double rangeMin, double rangeMax){
    double maxDiff = (average-rangeMin) >= (rangeMax-average) ? (average-rangeMin)  : (rangeMax-average);
    squashCoeff = radius/maxDiff;
    translationCoeff = center - squashCoeff * average;
}

double TransformLinear::transformFromPrice(double price){
    return (price * squashCoeff) + translationCoeff;
}

double TransformLinear::transformToPrice(double x){
    return (x - translationCoeff) / squashCoeff;
}

