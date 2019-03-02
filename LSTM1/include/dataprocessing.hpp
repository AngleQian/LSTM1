//
//  dataprocessing.hpp
//  LSTM1
//
//  Created by Angle Qian on 11/09/2018.
//  Copyright © 2018 Angle Qian. All rights reserved.
//

#ifndef dataprocessing_hpp
#define dataprocessing_hpp

#include <stdio.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>

namespace dataprocessing {
    const std::string baseDirectory = "/Users/angleqian/Drive Sync/Extended Essay/LSTM1/LSTM1/";
    
    std::vector< std::vector<std::string> > readCSV(std::string);
    std::vector< std::vector<double> > transformTableToDouble(std::vector< std::vector<std::string> >);
    
    template<typename T>
    std::vector< std::vector<T> >  extractColumnsFromTable(const std::vector<int>& columns, const std::vector< std::vector<T> >& inputTable) {
        std::vector< std::vector<T> > returnTable = std::vector< std::vector<T> >();
        std::vector<T> returnRow = std::vector<T>();
        
        for(std::vector<T> inputRow: inputTable){
            for(int column: columns) {
                returnRow.push_back(inputRow[column]);
            }
            returnTable.push_back(returnRow);
            returnRow.clear();
        }
        
        return returnTable;
    }
    
    double getTableColumnAverage(std::vector< std::vector<double> >, int);
    double getTableColumnMax(std::vector< std::vector<double> >, int);
    double getTableColumnMin(std::vector< std::vector<double> >, int);
    double getTableColumnSd(std::vector< std::vector<double> >, int);
}

#endif /* dataprocessing_hpp */
