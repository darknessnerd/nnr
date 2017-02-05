#include "FunctionApproximationNetwork.h"

FunctionApproximationNetwork::FunctionApproximationNetwork():NeuralNetwork(1)
{
}

FunctionApproximationNetwork::~FunctionApproximationNetwork()
{
}

void FunctionApproximationNetwork::init()
{
   nn::Layer *first_layer = new nn::Layer(this->numberInputs, 0);

   first_layer->addNeuron(nn::TransferFunctionType::LogSigType);
   first_layer->addNeuron(nn::TransferFunctionType::LogSigType);

   this->addLayer(first_layer);


   nn::Layer *second_layer = new nn::Layer(first_layer->getNumNeurons(), 1);

   second_layer->addNeuron(nn::TransferFunctionType::PureLinearType);

   this->addLayer(second_layer);

}
