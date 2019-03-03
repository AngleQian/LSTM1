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

std::vector< std::vector<std::string> > rawTable;
std::vector< std::vector<double> > processedTable;

void init();
void prepareData();
void prepareTest02();
void test02();
void test(int, Transform*, std::vector<int>, int, double, std::vector< std::vector<double> >, std::vector< std::vector<double> >, std::vector< std::vector<double> > , std::vector< std::vector<double> >);
void restateTesting();
void prepareSineTest();
void testSine();

namespace NetworkConfig {
    double alpha;
    std::vector<int> topology;
    int cellsPerBlock;
    std::vector< std::vector<double> > trainingInputs;
    std::vector< std::vector<double> > trainingOutputs;
    std::vector< std::vector<double> > validationInputs;
    std::vector< std::vector<double> > validationOutputs;
    Transform* transform;
};

int main(int argc, const char * argv[]) {
    init();
    
    
    
//    prepareData();
//    prepareTest02();
//    test02();
    
    prepareSineTest();
    restateTesting();
    testSine();
}

void test02() {
    std::ofstream file;
    std::string directory = "/Users/angleqian/Drive Sync/Extended Essay/LSTM1/LSTM1/output/param.txt";
    file.open(directory, std::ios_base::app);
    
    using namespace NetworkConfig;
    
//    for(double a = 0.2; a <= 20; a += 0.2) {
//        test(5, transform, topology, cellsPerBlock, a, trainingInputs, trainingOutputs, validationInputs, validationOutputs);
//    }
    test(10, transform, topology, cellsPerBlock, 10, trainingInputs, trainingOutputs, validationInputs, validationOutputs);
}


void prepareTest02() {
    int days = 60;
    std::vector< std::vector<double> > inputTable = std::vector< std::vector<double> >();
    std::vector< std::vector<double> > outputTable = std::vector< std::vector<double> >();
    unsigned long size = processedTable.size();
    
    std::vector<double> inputRow = std::vector<double>();
    for(int i = days; i < size - days; ++i) {
        inputRow.push_back((double) (processedTable[i][0] - processedTable[i - days][0]) / processedTable[i - days][0]);
        inputTable.push_back(inputRow);
        inputRow.clear();
    }
    
    std::vector<double> outputRow = std::vector<double>();
    for(int i = days; i < size - days; ++i) {
        outputRow.push_back((double) (processedTable[i + days][0] - processedTable[i][0]) / processedTable[i][0]);
        outputTable.push_back(outputRow);
        outputRow.clear();
    }
    
    unsigned long divider = 0.8 * size;
    
    using namespace NetworkConfig;
    
    trainingInputs = std::vector< std::vector<double> >(inputTable.begin(), inputTable.begin() + divider);
    trainingOutputs = std::vector< std::vector<double> >(outputTable.begin(), outputTable.begin() + divider);
    
    validationInputs = std::vector< std::vector<double> >(inputTable.begin() + divider + 1, inputTable.end());
    validationOutputs = std::vector< std::vector<double> >(outputTable.begin() + divider + 1, outputTable.end());
    
    std::cout << "Training Period: " << rawTable[days][0] << " - " << rawTable[divider][0]  << std::endl;
    std::cout << "Validation Period: " << rawTable[divider + 1][0] << " - " << rawTable[rawTable.size() - 1 - days][0] << std::endl;
    
    double average = dataprocessing::getTableColumnAverage(inputTable, 0);
    double sd = dataprocessing::getTableColumnSd(inputTable, 0);
    
    transform = new TransformStandardize(average, sd);
    
    topology = {1, 5, 1};
    cellsPerBlock = 1;
}

void prepareData() {
    // read file from CSV
    rawTable = dataprocessing::readCSV("data.csv");
    
    std::cout << "Time period: " << rawTable.front().at(0) << " - " << rawTable.back().at(0) << std::endl;
    
    // extract the needed the columns
    std::vector<int> neededColumns = {4};
    std::vector< std::vector<std::string> > filteredTable = dataprocessing::extractColumnsFromTable(neededColumns, rawTable);
    
    // convert table to all double type
    processedTable = dataprocessing::transformTableToDouble(filteredTable);
}

void restateTesting() {
    using namespace NetworkConfig;
    
    double sumOfError = 0;
    
    for(int i = 0; i < validationOutputs.size(); ++i) {
        sumOfError += (double) abs(validationOutputs[i][0] - validationInputs[i][0]);
    }
    
    std::cout << "Prediction via restating, mean absolute error: " << sumOfError / validationOutputs.size() << std::endl;
}

void test(int repetitions, Transform* transform, std::vector<int> topology, int cellsPerBlock, double alpha,
          std::vector< std::vector<double> > trainingInputs,
          std::vector< std::vector<double> > trainingOutputs, std::vector< std::vector<double> > validationInputs, std::vector< std::vector<double> > validationOutputs){
    std::ofstream file;
    std::string directory = "/Users/angleqian/Drive Sync/Extended Essay/LSTM1/LSTM1/output/param.txt";
    file.open(directory, std::ios_base::app);
    
    double sumOfError = 0;
    
    std::cout << std::endl << std::endl << "Alpha: " << alpha << std::endl;
    
    for(int i = 0; i != repetitions; ++i){
        Network network = Network(transform, topology, cellsPerBlock, alpha, trainingInputs, trainingOutputs, validationInputs, validationOutputs);
        network.train();
        sumOfError += network.validate();
        std::cout << i+1 << " ";
    }
    
    std::cout << "Average validation error: " << (double) sumOfError / repetitions << std::endl << std::endl;
    file << alpha << "," << (double) sumOfError / repetitions << "\n";
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

void prepareSineTest() {
    std::vector< std::vector<double> > inputTable = std::vector< std::vector<double> >();
    std::vector< std::vector<double> > outputTable = std::vector< std::vector<double> >();
    
    std::vector<double> inputRow = std::vector<double>();
    std::vector<double> outputRow = std::vector<double>();
    for(double i = 0; i <= 500; i += 0.1) {
        inputRow.push_back(0.5*cos(i));
        outputRow.push_back(0.5*cos((double) i + 2));
        inputTable.push_back(inputRow);
        outputTable.push_back(outputRow);
        inputRow.clear();
        outputRow.clear();
    }
    
    unsigned long size = inputTable.size();
    
    unsigned long divider = 0.8 * size;
    
    using namespace NetworkConfig;
    
    trainingInputs = std::vector< std::vector<double> >(inputTable.begin(), inputTable.begin() + divider);
    trainingOutputs = std::vector< std::vector<double> >(outputTable.begin(), outputTable.begin() + divider);
    
    validationInputs = std::vector< std::vector<double> >(inputTable.begin() + divider + 1, inputTable.end());
    validationOutputs = std::vector< std::vector<double> >(outputTable.begin() + divider + 1, outputTable.end());
    
    transform = new NoTransform();
    
    topology = {1, 5, 1};
    cellsPerBlock = 2;
}

void testSine() {
    std::ofstream file;
    std::string directory = "/Users/angleqian/Drive Sync/Extended Essay/LSTM1/LSTM1/output/param.txt";
    file.open(directory, std::ios_base::app);
    
    using namespace NetworkConfig;
    
    //    for(double a = 0.2; a <= 20; a += 0.2) {
    //        test(5, transform, topology, cellsPerBlock, a, trainingInputs, trainingOutputs, validationInputs, validationOutputs);
    //    }
    test(1, transform, topology, cellsPerBlock, 10, trainingInputs, trainingOutputs, validationInputs, validationOutputs);
}



