#ifndef LAYER_H
#define LAYER_H
#include "Neuron.h"
#include<vector>
using std::vector;
#include "Matrix.h"
using namespace math;

namespace nn
{
class Layer
{
public:
    /**  constructor
    *   \param layer's input number
    *   \param layer name
    */
    Layer(const unsigned int &numberInputs, const unsigned int &name);



    /** Default destructor */
    ~Layer();


    /** \brief
     *
     * return a matrix with
     *   - row = weight
     *   - column = neuron
     *
     * \param false bool transpose - transpose matrix
     * \return Matrix<double> - row = weight, column = neuron
     *
     */
    Matrix<double> getWeightMatrix(bool transpose = false);
    /** \brief
     * Get the bias matrix
     * \param false bool transpose
     * \return Matrix<double>
     *
     */
    Matrix<double> getBiasMatrix(bool transpose = false);
    void addNeuron(const TransferFunctionType &transferFunctionType);

    /**
    * \param layer inputs
    * \return layer result
    */
    vector<double> compute(const vector<double> &inputs, bool derivative = false);


    /** \brief
     * \return neuron numbers neurons
     */
    unsigned int getNumNeurons() const;

    unsigned int getNumInputs() const;




    /**
    * \param neuron index
    * \param input index
    * \param neuron weight to add
    * \return layer result
    */
    void addWeight(const unsigned int &neuron_index, const unsigned int &input_index,const double &weight);

    friend ostream &operator<<( ostream &output, const Layer *l )
    {
        output << "-------------- Layer name = [" << l->name << "] --------------\n";
        for(unsigned int i = 0; i < l->numNeurons; ++i)
            output << *l->neurons[i];
        return output;
    }
protected:
private:
    unsigned int numInputs = 0;
    unsigned int numNeurons = 0;

    unsigned int name = 0;

    Neuron **neurons = NULL;

};
}
#endif // LAYER_H
