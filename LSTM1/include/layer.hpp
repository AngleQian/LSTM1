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
    Layer();
    Layer(int, double);
    Layer(int, int, double, long);
    ~Layer();
    
    void forwardpass(const std::vector<double>&);
    void forwardpass(const std::shared_ptr<Layer>);
    void backwardpass(const std::shared_ptr<Layer>, const std::vector<double>&);
    void backwardpass(const std::shared_ptr<Layer>, const std::shared_ptr<Layer>);
    
    void flushState();
    void printLayer();
    
    std::shared_ptr<std::vector<double>> getOutput() const;
    std::shared_ptr<std::vector<std::shared_ptr<Unit>>> getUnits() { return std::make_shared<std::vector<std::shared_ptr<Unit>>>(units); }
private:
    std::vector<std::shared_ptr<Unit>> units;
};

#endif /* layer_hpp */
