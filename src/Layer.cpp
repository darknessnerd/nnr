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
void Layer::setWeights(const Matrix<double> &weights_matrix)
{
    if(weights_matrix.getRows() !=  this->numNeurons && weights_matrix.getColumns() != this->numInputs)
        throw std::invalid_argument("Layer::setWeights invalid weight matrix dimension");
    for(unsigned int neuron_index= 0; neuron_index < this->numNeurons; ++neuron_index)
    {
        for(unsigned int input_index= 0; input_index < this->numInputs; ++input_index)
        {
                double w = weights_matrix(neuron_index, input_index);
                this->neurons[neuron_index]->setWeight(input_index, w);
        }
    }
}
unsigned int Layer::getNumNeurons() const
{
    return this->numNeurons;
}

void Layer::setBiases(const Matrix<double> &bias)
{
    if(bias.getRows() !=  this->numNeurons && bias.getColumns() != 1)
        throw std::invalid_argument("Layer::setBiasMatrix invalid bias matrix dimension");

    for(unsigned int n = 0 ; n < this->numNeurons; ++n)
       this->neurons[n]->setBias(bias(n,0));

}

Matrix<double> Layer::getBiasMatrix(bool transpose)
{
//create the matrix, if transpose is true invert the rows and the colums
    double rows = this->numNeurons;
    double columns = 1;
    if(transpose)
    {
        rows = 1;
        columns = this->numNeurons;
    }
    Matrix<double> bias_matrix(rows, columns);
    //init the matrix with the waight values

        for(unsigned int n = 0 ; n < this->numNeurons; ++n)
        {

                double weight = this->neurons[n]->getBias();
                if(!transpose)
                    bias_matrix(n,0) = weight;
                else
                    bias_matrix(0,n) = weight;



        }

    return bias_matrix;
}
Matrix<double> Layer::getWeightMatrix(bool transpose )
{
    //create the matrix, if transpose is true invert the rows and the colums
    double rows = this->numNeurons;
    double columns = this->numInputs;
    if(transpose)
    {
        rows = this->numInputs;
        columns = this->numNeurons;
    }
    Matrix<double> weight_matrix(rows, columns);
    //init the matrix with the waight values
    for(unsigned int w = 0; w < this->numInputs; ++w)
    {
        for(unsigned int n = 0 ; n < this->numNeurons; ++n)
        {
            try
            {
                double weight = this->neurons[n]->getWeight(w);
                if(!transpose)
                    weight_matrix(n,w) = weight;
                else
                    weight_matrix(w,n) = weight;
            }
            catch (const std::invalid_argument& e)
            {
                std::rethrow_exception(std::current_exception());
            }


        }
    }
    return weight_matrix;
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
