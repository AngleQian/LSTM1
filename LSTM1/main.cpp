//
//  main.cpp
//  LSTM1
//
//  Created by Angle Qian on 11/09/2018.
//  Copyright © 2018 Angle Qian. All rights reserved.
//

#include <iostream>
#include <fstream>
#include <math.h>

#include "include/dataprocessing.hpp"
#include "include/utility.hpp"
#include "include/network.hpp"

void init(){
    std::ofstream file;
    std::string directory = "/Users/angleqian/Drive Sync/Extended Essay/LSTM1/LSTM1/output/debug1.txt";
    file.open(directory, std::ofstream::out | std::ofstream::trunc);
    file.close();
    directory = "/Users/angleqian/Drive Sync/Extended Essay/LSTM1/LSTM1/output/param.txt";
    file.open(directory, std::ofstream::out | std::ofstream::trunc);
    file.close();
}

void test(int repetitions, Transform* transform, std::vector<int> topology, int cellsPerBlock, double alpha,
          std::vector< std::vector<double> > trainingInputs,
          std::vector< std::vector<double> > trainingOutputs, std::vector< std::vector<double> > validationInputs, std::vector< std::vector<double> > validationOutputs){
    std::ofstream file;
    std::string directory = "/Users/angleqian/Drive Sync/Extended Essay/LSTM1/LSTM1/output/param.txt";
    file.open(directory, std::ios_base::app);
    
    double sumOfError = 0;
    
    for(int i = 0; i != repetitions; ++i){
        Network network = Network(transform, topology, cellsPerBlock, alpha, trainingInputs, trainingOutputs, validationInputs, validationOutputs);
        network.train();
        sumOfError += network.validate();
        
        std::cout << i << " ";
    }
    
    std::cout << std::endl << "Alpha: " << alpha << std::endl;
    std::cout << "Average validation error: " << (double) sumOfError / repetitions << std::endl << std::endl;
    file << alpha << "," << (double) sumOfError / repetitions << "\n";
}

int main(int argc, const char * argv[]) {
    init();
    
//    std::vector< std::vector<std::string> > rawTable = dataprocessing::readCSV("data.csv");
//
//    std::cout << rawTable.size() << std::endl;
//    
//    std::vector< std::vector<std::string> > filteredTable = std::vector< std::vector<std::string> >();
//    for(std::vector<std::string> rawRow : rawTable){
//        std::vector<std::string> filteredRow = std::vector<std::string>();
//        filteredRow.push_back(rawRow[4]); // get closing price
//        filteredTable.push_back(filteredRow);
//    }
//
//    std::vector< std::vector<double> > convertedTable = dataprocessing::transformTableToDouble(filteredTable);
//   std::vector< std::vector<double> > processedTable = dataprocessing::processTable(convertedTable);
//
//    long size = filteredTable.size();
//
//    std::vector< std::vector<double> > trainingInputs = dataprocessing::transformTableToDouble(std::vector< std::vector<std::string> >(filteredTable.begin(), filteredTable.begin() + round(0.8*size)));
//    
//    std::vector< std::vector<double> > validationInputs = dataprocessing::transformTableToDouble(std::vector< std::vector<std::string> >(filteredTable.begin() + round(0.8*size) + 1, filteredTable.end()));
//    
//    std::vector< std::vector<double> > trainingOutputs = std::vector< std::vector<double> >(trainingInputs.begin()+1, trainingInputs.end());
//    std::vector< std::vector<double> > validationOutputs = std::vector< std::vector<double> >(validationInputs.begin()+1, validationInputs.end());
//
//    double average = dataprocessing::getTableColumnAverage(trainingInputs, 0);
//    double min = dataprocessing::getTableColumnMin(trainingInputs, 0);
//    double max = dataprocessing::getTableColumnMax(trainingInputs, 0);
//    TransformLinear transform = TransformLinear(average, min, max);
//
//    std::cout << "Training Period: " << rawTable[0][0] << " - " << rawTable[trainingInputs.size()][0] << std::endl;
//    std::cout << "Validation Period: " << rawTable[trainingInputs.size()+1][0] << " - " << rawTable[size-1][0] << std::endl << std::endl;
    
    
    std::vector< std::vector<double > > sine = std::vector< std::vector<double > >();
    for(double i = 0.1; i <= 2000; i += 0.1){
        std::vector<double> value = std::vector<double>();
        value.push_back(0.1*sin(i));

        sine.push_back(value);
    }
    std::vector< std::vector<double > > trainingInputs = std::vector< std::vector<double> >(sine.begin(), sine.begin() + round(0.8 * sine.size()));
    std::vector< std::vector<double > > validationInputs = std::vector< std::vector<double> >(sine.begin() + round(0.8 * sine.size()) + 1, sine.end());
    
    std::vector<double> value;
    std::vector< std::vector<double > > trainingOutputs = std::vector< std::vector<double> >();
    for(int i = 1; i != trainingInputs.size(); ++i){
        value = std::vector<double>();
        value.push_back(trainingInputs[i][0]);
        trainingOutputs.push_back(value);
    }
    
    value = std::vector<double>();
    value.push_back(validationInputs[0][0]);
    trainingOutputs.push_back(value);
    
    std::vector< std::vector<double > > validationOutputs = std::vector< std::vector<double> >();
    for(int i = 1; i != validationInputs.size(); ++i){
        value = std::vector<double>();
        value.push_back(validationInputs[i][0]);
        validationOutputs.push_back(value);
    }    

    NoTransform transform = NoTransform();
    
    std::vector<int> topology = {1, 10, 1};
    int cellsPerBlock = 1;
    double alpha = 6;

    test(5, &transform, topology, cellsPerBlock, alpha, trainingInputs, trainingOutputs, validationInputs, validationOutputs);
}
