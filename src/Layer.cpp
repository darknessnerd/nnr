#include "Layer.h"
#include <stdexcept>
using namespace nn;
Layer::Layer(const unsigned int &numberInputs, const unsigned int &name):numInputs(numberInputs),name(name)
{

}

Layer::~Layer()
{
    std::cout << "~Layer= " << name << "\n";
    for(unsigned int i = 0 ; i < this->numNeurons; ++i)
    {
       std::cout << "~neuron= " << i << "\n";
        delete neurons[i];
    }
    delete[] neurons;
    std::cout << "delete Layer name= " << name << "\n";
}
void Layer::addWeight(const unsigned int &neuron_index,const unsigned int &input_index, const double &weight)
{
    if(neuron_index >= this->numNeurons)
       throw std::invalid_argument("Layer::addWeight invalid neuron index");
    if(input_index >= this->numInputs)
       throw std::invalid_argument("Layer::addWeight invalid input index");
    this->neurons[neuron_index]->addWeight(input_index, weight);
}
unsigned int Layer::getNumNeurons() const
{
    return this->numNeurons;
}


unsigned int Layer::getNumInputs() const
{
    return this->numInputs;
}

vector<double> Layer::compute(const vector<double> &inputs, bool derivative)
{

    vector<double> result(this->numNeurons, 0);

    for(unsigned int neuron = 0; neuron < this->numNeurons; ++neuron)
    {
           result[neuron] += this->neurons[neuron]->compute(inputs, derivative);
    }


    return result;
}


void Layer::addNeuron(const TransferFunctionType &transferFunctionType)
{
    //create the new neuron
    Neuron *neuron = new Neuron(numNeurons, this->numInputs, transferFunctionType);

    //allocate the new array of pointers
    Neuron **neurons_tmp = new Neuron*[++numNeurons]();
    //copy the  pointer in the new array
    for(unsigned int i = 0; i < numNeurons-1; ++i)
        neurons_tmp[i] = neurons[i];
    neurons_tmp[numNeurons-1] = neuron;

    delete []neurons;
    neurons = neurons_tmp;
}
