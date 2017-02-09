#include "Neuron.h"
using namespace nn;
#include<stdexcept>
#include<assert.h>

#include <algorithm>


Neuron::Neuron()
{
    this->w = nullptr;
    this->bias = 0;
    this->transferFunction = nullptr;
}
Neuron::Neuron(const unsigned int &index, const int &numberInputs, const TransferFunctionType &transferFunctionType):index(index),numberInputs(numberInputs)
{

    this->w = new double[this->numberInputs];
    this->bias = 0;
    switch (transferFunctionType)
    {
    case TransferFunctionType::HardLimitType:
        this->transferFunction = new HardLimit();
        break;
    case TransferFunctionType::PureLinearType:
        this->transferFunction = new PureLinear();
        break;
    case TransferFunctionType::HardLimitSymmetricalType:
        this->transferFunction = new HardLimitSymmetrical();
        break;
    case TransferFunctionType::LogSigType:
        this->transferFunction = new LogSigmoid();
        break;
    default:
        //TODO
        throw TransferFunctionTypeException();
    };
}

Neuron::~Neuron()
{
    delete transferFunction;
    delete[] w;
}

double Neuron::getWeight(const unsigned int weight_index) const
{
    if(weight_index >= numberInputs)
        throw std::invalid_argument("Neuron::getWeight invalid weight index");
    return w[weight_index];
}
double Neuron::productory(const vector<double> &inputs)
{
    double productory = 0;
    unsigned int i = 0;
    for(vector<double>::const_iterator it = inputs.begin(); it != inputs.end(); ++it)
    {
        productory += (*it) * this->w[i++];
    }
    return productory + this->bias;
}

double Neuron::getBias() const
{
    return bias;
}
void Neuron::setBias(const double bias_value)
{
    this->bias = bias_value;
}
double Neuron::compute(const vector<double> &inputs, bool derivate )
{
    double productory = this->productory(inputs);
    if(!derivate)
        return this->transferFunction->compute(productory);
    return this->transferFunction->derivative(productory);
}
Neuron::Neuron(const Neuron& other)
{
    std::cout << "Neuron::Neuron copy constructor\n";
    copyFrom(other);
}

void Neuron::copyFrom(const Neuron& other)
{
    this->bias = other.bias;
    this->numberInputs = other.numberInputs;
    this->index = other.index;

    this->w = new double[this->numberInputs];
    for(unsigned int i = 0 ; i < this->numberInputs; ++i)
        this->w[i] = other.w[i];


    if(dynamic_cast<const HardLimit*>(other.transferFunction) != nullptr)
        this->transferFunction = new HardLimit();
    else if(dynamic_cast<const PureLinear*>(other.transferFunction) != nullptr)
        this->transferFunction = new PureLinear();
    else if(dynamic_cast<const HardLimitSymmetrical*>(other.transferFunction) != nullptr)
        this->transferFunction = new HardLimitSymmetrical();
    else if(dynamic_cast<const LogSigmoid*>(other.transferFunction) != nullptr)
        this->transferFunction = new LogSigmoid();
    else
        throw TransferFunctionTypeException();

}
void Neuron::addWeight(const unsigned int input_index, const double weight)
{

    if(input_index >= this->numberInputs)
        throw std::invalid_argument("Neuron::addWeight invalid input index");

    this->w[input_index]+=weight;

}
void Neuron::setWeight(const unsigned int input_index, const double weight)
{

    if(input_index >= this->numberInputs)
        throw std::invalid_argument("Neuron::setWeight invalid input index");

    this->w[input_index] =weight;

}


int Neuron::getNumberInputs() const
{
    return this->numberInputs;
}
/**
* copy assignment operator
*/
Neuron& Neuron::operator=(const Neuron& rhs)
{
    std::cout << "Neuron::operator=(const Neuron& rhs)\n";
    if (this == &rhs) return *this; // handle self assignment
    delete transferFunction;
    delete[] w;
    copyFrom(rhs);
    //assignment operator
    return *this;
}
/**
* Move assignment operator std::move()
*/
Neuron& Neuron::operator=(Neuron&& other) // move assignment
{
    std::cout << "Neuron::operator=(Neuron&& other) operator=\n";
    assert(this != &other); // self-assignment check not required
    // delete this storage
    delete[] this->w;
    delete this->transferFunction;
    copyFrom(other);
    other.w = nullptr;
    other.transferFunction = nullptr;
    other.bias = 0;
    other.index = 0;
    other.numberInputs = 0;
    return *this;
}





