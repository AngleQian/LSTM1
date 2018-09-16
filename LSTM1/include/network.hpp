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

class Network {
public:
    Network(Transform*, std::vector<int>, int, std::vector< std::vector<double> >,
            std::vector< std::vector<double> >);
    
    void weightInit();
    
    void train();
    void fowardpass();
    void backwardpass();
    void validation();
    
    void printNetwork();
    
    std::vector<std::shared_ptr<Layer>>* getLayers() { return &layers; }
private:
    Transform* transform;
    std::vector<std::shared_ptr<Layer>> layers;
    std::vector< std::vector<double> > trainingSet;
    std::vector< std::vector<double> > validationSet;
};

#endif /* network_hpp */
