//
//  utility.cpp
//  LSTM1
//
//  Created by Angle Qian on 12/09/2018.
//  Copyright © 2018 Angle Qian. All rights reserved.
//

#include "../include/utility.hpp"

double utility::f(double x){
    return (double) 1 / (1+exp(-1*x));
}

double utility::df(double x){
    return (double) f(x) * (1-f(x));
}

double utility::g(double x){
    return (double) 4 / (1 + exp(-1*x)) - 2;
}

double utility::dg(double x){
    return (double) 4 * df(x);
}

double utility::h(double x){
    return (double) 2 / (1+exp(-1*x)) - 1;
}

double utility::dh(double x){
    return (double) 2 * df(x);
}

double utility::nofunc(double x){
    return x;
}

double utility::dnofunc(double x){
    return 1.0;
}

double utility::relu(double x){
    return x > 0 ? x : 0;
}

double utility::drelu(double x){
    return x > 0 ? 1 : 0;
}

double utility::leakyrelu(double x){
    return x > 0 ? x : 0.01 * x;
}

double utility::dleakyrelu(double x){
    return x > 0 ? 1 : 0.01;
}


double utility::tanh(double x) {
    return (double) 2 / (1 + exp(-2 * x)) - 1;
}

double utility::dtanh(double x) {
    return (double) 1 - pow(tanh(x), 2.0);
}

double utility::getError(double testValue, double truthValue) {
    return getAbsoluteError(testValue, truthValue);
}

double utility::getAbsoluteError(double testValue, double truthValue) {
    return abs((double) testValue - truthValue);
}

double utility::getRelativeError(double testValue, double truthValue) {
    return abs((double) testValue - truthValue) / truthValue;
}

double utility::clipping(double x){
    if (x > 1) {
        return 1;
    }
    
    if (x < -1) {
        return -1;
    }
    return x;
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

TransformStandardize::TransformStandardize(double mean, double sd){
    this->mean = mean;
    this->sd = sd;
}

double TransformStandardize::transformFromPrice(double price) {
    return (double) (price - mean) / ((double) sd);
}

double TransformStandardize::transformToPrice(double x) {
    return (double) x * sd + mean;
}

