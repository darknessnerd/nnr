#ifndef EXAMPLE_H
#define EXAMPLE_H

#include "NeuralNetwork.h"


/**
   3 layer networs
**/
class Network3Layer: public nn::NeuralNetwork
{
public:
    Network3Layer():nn::NeuralNetwork(3)
    {
    };
    ~Network3Layer()
    {
        std::cout << "~Network3Layer\n";
    }
    void init()
    {
        nn::Layer *layer = new nn::Layer(this->numberInputs, 0);
        layer->addNeuron(nn::TransferFunctionType::HardLimitType);
        layer->addNeuron(nn::TransferFunctionType::HardLimitType);
        this->addLayer(layer);

        layer = new nn::Layer(2, 1);
        layer->addNeuron(nn::TransferFunctionType::HardLimitType);
        layer->addNeuron(nn::TransferFunctionType::HardLimitType);
        this->addLayer(layer);

        layer = new nn::Layer(2, 2);
        layer->addNeuron(nn::TransferFunctionType::HardLimitType);
        layer->addNeuron(nn::TransferFunctionType::HardLimitType);
        this->addLayer(layer);
    };
};



/**
  The linear associator is an example of a type of neural network called an
associative memory. The task of an associative memory is to learn pairs
of prototype input/output vectors:
{{p1, t1}, { p2, t2 } , ..., {pq , tq}}
In other words, if the network receives an input then it should produce an output , for . In addition, if the input is
changed slightly (i.e., ) then the output should only be changed
slightly (i.e., )
**/
class LinearAssociator: public nn::NeuralNetwork
{
private:
   unsigned int numNeurons;
public:
    LinearAssociator(const unsigned int &numInputs):nn::NeuralNetwork(numInputs),numNeurons(numInputs)
    {
    };
    ~LinearAssociator()
    {
        std::cout << "~LinearAssociator\n";
    }
    void init()
    {
        nn::Layer *layer = new nn::Layer(this->numberInputs, 0);
        for(unsigned int i = 0; i < this->numNeurons; ++i)
           layer->addNeuron(nn::TransferFunctionType::PureLinearType);
        this->addLayer(layer);
    };
};

#endif // EXAMPLE_H
