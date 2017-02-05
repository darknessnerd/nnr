#include "Perceptron.h"

Perceptron::Perceptron(const unsigned int &numInputs, const unsigned int &numNeurons):nn::NeuralNetwork(numInputs),numNeurons(numNeurons)
{
}
Perceptron::~Perceptron()
{
    std::cout << "~Perceptron\n";
}
void Perceptron::init()
{
    nn::Layer *layer = new nn::Layer(this->numberInputs, 0);
    for(unsigned int i = 0; i < this->numNeurons; ++i)
        layer->addNeuron(nn::TransferFunctionType::HardLimitType);
    this->addLayer(layer);
}
