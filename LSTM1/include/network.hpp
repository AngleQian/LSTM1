//
//  network.hpp
//  LSTM1
//
//  Created by Angle Qian on 11/09/2018.
//  Copyright © 2018 Angle Qian. All rights reserved.
//

#ifndef network_hpp
#define network_hpp

#include "layer.hpp"
#include "dataprocessing.hpp"

class Network {
public:
    Network(Transform*, std::vector<int>, int, double, std::vector< std::vector<double> >,
            std::vector< std::vector<double> >, std::vector< std::vector<double> >, std::vector< std::vector<double> >);
    void weightInit(int);
    void train();
    std::vector<double> forwardpass(const std::vector<double>&);
    double backwardpass(const std::vector<double>&, const std::vector<double>&);
    double validate();
    
    void flushState();
    void printNetwork();
    std::shared_ptr<std::vector<std::shared_ptr<Layer>>> getLayers() { return std::make_shared<std::vector<std::shared_ptr<Layer>>>(layers); }
    
private:
    Transform* transform;
    double alpha;
    std::vector<std::shared_ptr<Layer>> layers;
    std::vector< std::vector<double> > trainingInputs;
    std::vector< std::vector<double> > validationInputs;
    std::vector< std::vector<double> > trainingOutputs;
    std::vector< std::vector<double> > validationOutputs;
};

#endif /* network_hpp */
