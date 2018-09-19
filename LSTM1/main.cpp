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
    std::vector< std::vector<std::string> > rawTable = dataprocessing::readCSV("data.csv");
    
    std::vector< std::vector<std::string> > filteredTable = std::vector< std::vector<std::string> >();
    for(std::vector<std::string> rawRow : rawTable){
        std::vector<std::string> filteredRow = std::vector<std::string>();
        filteredRow.push_back(rawRow[4]); // get closing price
        filteredTable.push_back(filteredRow);
    }
    
    long size = filteredTable.size();
    
    std::vector< std::vector<double> > trainingInputs = dataprocessing::transformTableToDouble(std::vector< std::vector<std::string> >(filteredTable.begin(), filteredTable.begin() + round(0.8*size)));
    
    std::vector< std::vector<double> > validationInputs = dataprocessing::transformTableToDouble(std::vector< std::vector<std::string> >(filteredTable.begin() + round(0.8*size) + 1, filteredTable.end()));
    
    std::vector< std::vector<double> > trainingOutputs = trainingInputs;
    std::vector< std::vector<double> > validationOutputs = validationInputs;
    
    double average = dataprocessing::getTableColumnAverage(trainingInputs, 0);
    double min = dataprocessing::getTableColumnMin(trainingInputs, 0);
    double max = dataprocessing::getTableColumnMax(trainingInputs, 0);
    TransformLinear transform = TransformLinear(average, min, max);
    
    std::vector<int> topology = {1, 2, 3, 2, 1};
    int cellsPerBlock = 2;
    
    Network network = Network(&transform, topology, cellsPerBlock, trainingInputs, validationInputs, trainingOutputs, validationOutputs);

    network.printNetwork();
    network.train();
}
