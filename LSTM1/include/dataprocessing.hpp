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

namespace dataprocessing {
    const std::string baseDirectory = "/Users/angleqian/Drive Sync/Extended Essay/LSTM1/LSTM1/data/";
    
    std::vector< std::vector<std::string> > readCSV(std::string);
    std::vector< std::vector<double> > transformTableToDouble(std::vector< std::vector<std::string> >);
    double getTableColumnAverage(std::vector< std::vector<std::string> >, int);
    double getTableColumnMax(std::vector< std::vector<std::string> >, int);
    double getTableColumnMin(std::vector< std::vector<std::string> >, int);
    void printTable(std::vector< std::vector<std::string> >);
}

#endif /* dataprocessing_hpp */
