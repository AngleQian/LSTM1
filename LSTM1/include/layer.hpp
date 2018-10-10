//
//  layer.hpp
//  LSTM1
//
//  Created by Angle Qian on 11/09/2018.
//  Copyright © 2018 Angle Qian. All rights reserved.
//

#ifndef layer_hpp
#define layer_hpp

#include "memoryblock.hpp"
#include "neuron.hpp"

class Layer{
public:
    Layer(); // base constructor
    Layer(int, double); // for regular neuron layers
    Layer(int, int, double, long); // for LSTM layers
    
    void forwardpass(const std::vector<double>&);
    void forwardpass(const std::shared_ptr<Layer>);
    
    void backwardpass(const std::shared_ptr<Layer>, const std::vector<double>&);
    void backwardpass(const std::shared_ptr<Layer>, const std::shared_ptr<Layer>);
    
    std::vector<double>* getOutput() const;
    
    void printLayer();
    
    std::vector<std::shared_ptr<Unit>>* getUnits() { return &units; }
private:
    std::vector<std::shared_ptr<Unit>> units;
};

#endif /* layer_hpp */
