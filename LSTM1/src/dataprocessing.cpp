//
//  dataprocessing.cpp
//  LSTM1
//
//  Created by Angle Qian on 11/09/2018.
//  Copyright © 2018 Angle Qian. All rights reserved.
//

#include "../include/dataprocessing.hpp"

std::vector< std::vector<std::string> > dataprocessing::readCSV(std::string filename){
    std::vector< std::vector<std::string> > ret = std::vector< std::vector<std::string> >();
    
    std::string path = dataprocessing::baseDirectory + "/" + filename;
    
    std::ifstream csv(path);
    std::string line;
    
    while(std::getline(csv, line)){
        std::stringstream lineStream(line);
        std::string cell;
        
        std::vector<std::string> ret_inner = std::vector<std::string>();
        
        while(std::getline(lineStream, cell, ',')){
            ret_inner.push_back(cell);
        }
        
        ret.push_back(ret_inner);
    }
    
    return ret;
}

std::vector< std::vector<double> > dataprocessing::transformTableToDouble(std::vector<std::vector<std::string> > rawTable){
    std::vector< std::vector<double> > table = std::vector< std::vector<double> >();
    for(std::vector<std::string> rawRow : rawTable){
        std::vector<double> row = std::vector<double>();
        
        for(std::string rawCell : rawRow){
            row.push_back(stod(rawCell));
        }
        
        table.push_back(row);
    }
    
    return table;
}

double dataprocessing::getTableColumnAverage(std::vector< std::vector<double> > table, int columnNumber){
    long double sum = 0;
    int count = 0;
    
    for(std::vector<double> row : table) {
        sum += row[columnNumber];
        ++count;
    }
    
    return sum / count;
}

double dataprocessing::getTableColumnMax(std::vector<std::vector<double> > table, int columnNumber){
    double max = table[1][1];
    for(std::vector<double> row : table) {
        if (max < row[columnNumber]){
            max = row[columnNumber];
        }
    }
    return max;
}

double dataprocessing::getTableColumnMin(std::vector<std::vector<double> > table, int columnNumber){
    double min = table[1][1];
    for(std::vector<double> row : table) {
        if (min > row[columnNumber]){
            min = row[columnNumber];
        }
    }
    return min;
}

