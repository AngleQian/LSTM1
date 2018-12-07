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

void init();
void test(int, Transform*, std::vector<int>, int, double, std::vector< std::vector<double> >, std::vector< std::vector<double> >, std::vector< std::vector<double> > , std::vector< std::vector<double> >);
void restate(std::vector< std::vector<double> >, std::vector< std::vector<double> >);

int main(int argc, const char * argv[]) {
    init();
//
//    std::vector< std::vector<std::string> > rawTable = dataprocessing::readCSV("data.csv");
//
//    std::vector< std::vector<std::string> > filteredTable = std::vector< std::vector<std::string> >();
//    for(std::vector<std::string> rawRow : rawTable){
//        std::vector<std::string> filteredRow = std::vector<std::string>();
//        filteredRow.push_back(rawRow[4]); // get closing price
//        filteredTable.push_back(filteredRow);
//    }
//
////    std::vector< std::vector<double> > convertedTable = dataprocessing::transformTableToDouble(filteredTable);
////   std::vector< std::vector<double> > processedTable = dataprocessing::processTable(convertedTable);
//
//    long size = filteredTable.size();
//
//    std::vector< std::vector<double> > table_ = dataprocessing::transformTableToDouble(filteredTable);
//    std::vector< std::vector<double> > table = std::vector< std::vector<double> >(table_.begin(), table_.end());
//
//    size = table.size();
//
//    std::vector<double> temp;
//
//    std::vector< std::vector<double> > trainingInputs = std::vector< std::vector<double> >();
//    for(int i = 2; i != round(0.8*size); ++i){
//        temp = std::vector<double>();
//        temp.push_back(table[i][0]);
//        temp.push_back(table[i-1][0]);
//        temp.push_back(table[i-2][0]);
//        trainingInputs.push_back(temp);
//    }
//
//    std::vector< std::vector<double> > validationInputs = std::vector< std::vector<double> >();
//    for(int i = round(0.8*size) + 2; i != size; ++i){
//        temp = std::vector<double>();
//        temp.push_back(table[i][0]);
//        temp.push_back(table[i-1][0]);
//        temp.push_back(table[i-2][0]);
//        validationInputs.push_back(temp);
//    }
//
//    std::vector< std::vector<double> > trainingOutputs = std::vector< std::vector<double> >();
//    for(int i = 1; i != trainingInputs.size(); ++i){
//        temp = std::vector<double>();
//        temp.push_back(trainingInputs[i][0]);
//        trainingOutputs.push_back(temp);
//    }
//
//    std::vector< std::vector<double> > validationOutputs = std::vector< std::vector<double> >();
//    for(int i = 1; i != validationInputs.size(); ++i){
//        temp = std::vector<double>();
//        temp.push_back(validationInputs[i][0]);
//        validationOutputs.push_back(temp);
//    }
//
//    double average = dataprocessing::getTableColumnAverage(table, 0);
//    double min = dataprocessing::getTableColumnMin(table, 0);
//    double max = dataprocessing::getTableColumnMax(table, 0);
//    TransformLinear transform = TransformLinear(average, min, max);
//
//    std::cout << "Training Period: " << rawTable[0][0] << " - " << rawTable[trainingInputs.size()][0] << std::endl;
//    std::cout << "Validation Period: " << rawTable[trainingInputs.size()+1][0] << " - " << rawTable[size-1][0] << std::endl << std::endl;
    
    
    std::vector< std::vector<double > > sine = std::vector< std::vector<double > >();
    for(double i = 0.1; i <= 250; i += 0.1){
        std::vector<double> value = std::vector<double>();
        value.push_back(0.25*sin(i));

        sine.push_back(value);
    }
    std::vector< std::vector<double > > trainingInputs = std::vector< std::vector<double> >(sine.begin(), sine.begin() + round(0.8 * sine.size()));
    std::vector< std::vector<double > > validationInputs = std::vector< std::vector<double> >(sine.begin() + round(0.8 * sine.size()) + 1, sine.end() - 1);

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
    
    value = std::vector<double>();
    value.push_back(sine[sine.size() - 1][0]);
    validationOutputs.push_back(value);

    NoTransform transform = NoTransform();
    
    std::vector<int> topology = {1, 5, 1};
    int cellsPerBlock = 1;
    double alpha = 5;

//    for(alpha = 2; alpha <= 20; alpha += 0.2){
//        test(20, &transform, topology, cellsPerBlock, alpha, trainingInputs, trainingOutputs, validationInputs, validationOutputs);
//    }
    test(1, &transform, topology, cellsPerBlock, alpha, trainingInputs, trainingOutputs, validationInputs, validationOutputs);
    
}

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
    
    std::cout << std::endl << "Alpha: " << alpha << std::endl;
    
    for(int i = 0; i != repetitions; ++i){
        Network network = Network(transform, topology, cellsPerBlock, alpha, trainingInputs, trainingOutputs, validationInputs, validationOutputs);
        network.train();
        sumOfError += network.validate();
        std::cout << i+1 << " ";
    }
    
    std::cout << std::endl << "Average validation error: " << (double) sumOfError / repetitions << std::endl << std::endl;
    file << alpha << "," << (double) sumOfError / repetitions << "\n";
    
    restate(validationInputs, validationOutputs);
}

void restate(std::vector< std::vector<double> > validationInputs, std::vector< std::vector<double> > validationOutputs){
    double sumOfError = 0;
    
    for(int i = 0; i != validationOutputs.size(); ++i){
        sumOfError += (double) abs(validationOutputs[i][0] - validationInputs[i][0]) / validationOutputs[i][0];
    }
    
    std::cout << "Prediction via restating: " << std::endl;
    std::cout << "Average error: " << (double) sumOfError / validationOutputs.size() << std::endl << std::endl;
}
