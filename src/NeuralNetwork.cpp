#include "NeuralNetwork.h"
using namespace nn;
NeuralNetwork::NeuralNetwork(const unsigned int numberInputs):numberInputs(numberInputs)
{
    cout << "Construct network!\n" ;

}

NeuralNetwork::~NeuralNetwork()
{
    std::cout << "~NeuralNetwork" << std::endl;
    for(unsigned int i = 0; i < numLayers; ++i)
        delete layers[i];
    delete []layers;

}
unsigned int NeuralNetwork::getNumberInputs() const
{
    return this->numberInputs;
}
unsigned int NeuralNetwork::getNumberOutputs() const
{
    return this->numberOutputs;
}

void NeuralNetwork::addWeight(const unsigned int layer_index,const unsigned int neuron_index, const unsigned int input_index, const double weight)
{

    if(layer_index >= this->numLayers)
        throw std::invalid_argument("NeuralNetwork::addWeight invalid layer index");

    try
    {
        this->layers[layer_index]->addWeight(neuron_index,input_index, weight);
    }
    catch (const std::invalid_argument& e)
    {
        std::rethrow_exception(std::current_exception());
    }

}

void NeuralNetwork::setWeights(const unsigned int layer_index, const Matrix<double> & weight_matrix)
{
   if(layer_index >= this->numLayers)
        throw std::invalid_argument("NeuralNetwork::setWeights invalid layer index");

    try
    {
        this->layers[layer_index]->setWeights( weight_matrix);
    }
    catch (const std::invalid_argument& e)
    {
        std::rethrow_exception(std::current_exception());
    }


}
void NeuralNetwork::setBiases(const unsigned int layer_index, const Matrix<double> & bias_matrix)
{
   if(layer_index >= this->numLayers)
        throw std::invalid_argument("NeuralNetwork::setBiases invalid layer index");

    try
    {
        this->layers[layer_index]->setBiases( bias_matrix);
    }
    catch (const std::invalid_argument& e)
    {
        std::rethrow_exception(std::current_exception());
    }


}

Matrix<double> NeuralNetwork::getBiasMatrixOfLayer(const unsigned int layer_index, bool transpose ) const
{
    if(layer_index >= this->numLayers)
        throw std::invalid_argument("NeuralNetwork::getBiasMatrixOfLayer invalid layer index");


    try
    {
        return this->layers[layer_index]->getBiasMatrix(transpose);
    }
    catch (const std::invalid_argument& e)
    {
        std::rethrow_exception(std::current_exception());
    }

    return Matrix<double>(0,0);
}
Matrix<double> NeuralNetwork::getWeightMatrixOfLayer(const unsigned int layer_index, bool transpose ) const
{
    if(layer_index >= this->numLayers)
        throw std::invalid_argument("NeuralNetwork::getWeightMatrixOfLayer invalid layer index");


    try
    {
        return this->layers[layer_index]->getWeightMatrix(transpose);
    }
    catch (const std::invalid_argument& e)
    {
        std::rethrow_exception(std::current_exception());
    }

    return Matrix<double>(0,0);

}


vector<double> NeuralNetwork::compute(const vector<double> &inputs, bool derivative)
{

    if(inputs.size() != this->numberInputs)
    {
        throw std::invalid_argument("NeuralNetwork::compute invalid inputs size");
    }


    vector<double> result(inputs);
    for(unsigned int layer = 0; layer < this->numLayers; ++layer)
    {
        result = this->layers[layer]->compute(result, derivative );
    }
    return result;
}
vector<double> NeuralNetwork::computeOutputLayer(const unsigned int layer_index, const vector<double> &inputs, bool derivative)
{
     if(layer_index >= this->numLayers && layer_index < 0)
        throw std::invalid_argument("NeuralNetwork::computeOutputLayers invalid layer index");

    try
    {
        vector<double> result(inputs);
        for(unsigned int layer = 0; layer <= layer_index; ++layer)
        {
            result = this->layers[layer]->compute(result,derivative);
        }
        return result;
    }
    catch (const std::invalid_argument& e)
    {
        std::rethrow_exception(std::current_exception());
    }
    return vector<double>();
}



unsigned int NeuralNetwork::getNumLayers() const
{
  return this->numLayers;
}


void NeuralNetwork::random_weight()
{
    unsigned int numLayer = this->getNumLayers();
    //random weight
    for(unsigned int l = 0 ; l <  numLayer; ++l)
    {

        unsigned int numNeurons = this->layers[l]->getNumNeurons();
        unsigned int layerNumInputs = this->layers[l]->getNumInputs();


        for(unsigned int n = 0 ;  n < numNeurons; ++n)
        {
            for(unsigned int i = 0 ;  i < layerNumInputs; ++i)
            {
                this->addWeight(l, n,i,0.5);
            }

        }
    }
}

void NeuralNetwork::addLayer(Layer *layer)
{

    //allocate the new array of pointers
    Layer **layers_tmp = new Layer*[++numLayers]();
    //copy the reference pointer in the new array
    for(unsigned int i = 0; i < numLayers-1; ++i)
        layers_tmp[i] = layers[i];
    layers_tmp[numLayers-1] = layer;
    this->numberOutputs = layer->getNumNeurons();
    delete []layers;
    layers = layers_tmp;
}
