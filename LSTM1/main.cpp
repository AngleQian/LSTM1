//
//  main.cpp
//  LSTM1
//
//  Created by Angle Qian on 11/09/2018.
//  Copyright © 2018 Angle Qian. All rights reserved.
//

#include <iostream>
#include <math.h>

#include "include/dataprocessing.hpp"
#include "include/utility.hpp"
#include "include/network.hpp"

int main(int argc, const char * argv[]) {    
    std::vector< std::vector<std::string> > table = dataprocessing::readCSV("data.csv");
    
    // dataprocessing::printTable(table);
    
    double average = dataprocessing::getTableColumnAverage(table, 1);
    double min = dataprocessing::getTableColumnMin(table, 1);
    double max = dataprocessing::getTableColumnMax(table, 1);
    TransformLinear transform = TransformLinear(average, min, max);
    
    std::vector<int> topology = {1, 2, 3, 3, 2, 1};
    int cellsPerBlock = 2;
    long size = table.size();
    
    std::vector< std::vector<double> > trainingSet = dataprocessing::transformTableToDouble(std::vector< std::vector<std::string> >(table.begin(), table.begin() + round(0.8*size)));
    
    std::vector< std::vector<double> > validationSet = dataprocessing::transformTableToDouble(std::vector< std::vector<std::string> >(table.begin() + round(0.8*size) + 1, table.end()));
    
    Network network = Network(&transform, topology, cellsPerBlock, trainingSet, validationSet);
    
    network.printNetwork();
}
