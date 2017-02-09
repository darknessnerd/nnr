#ifndef NEURON_H
#define NEURON_H
#include <vector>
using std::vector;
#include <iostream>
using std::ostream;
using std::cout;
#include "TransferFunction.h"
namespace nn
{


class Neuron
{

protected:
private:
    double bias;
    double *w; //weights
    TransferFunction* transferFunction = NULL;
    unsigned int index;
    unsigned int numberInputs;

    void copyFrom(const Neuron& other);

    double productory(const vector<double> &inputs);
public:
    /**
    * Empty construcotr
    */
    Neuron();
    /** constructor
    *   \param neuron index in the layer
    *   \param transfer function of the neuron
    *
    */
    Neuron(const unsigned int &index, const int &numberInputs, const TransferFunctionType &transferFunctionType);
    /** Default destructor */
    virtual ~Neuron();
    /** Copy constructor
     *  \param other Object to copy from
     */
    Neuron(const Neuron& other);
    /** Assignment operator
     *  \param other Object to assign from
     *  \return A reference to this
     */
    Neuron& operator=(const Neuron& other);

    /**
    * Move assignment operator
    */
    Neuron& operator=(Neuron&& other);

    int getNumberInputs() const;

    void addWeight(const unsigned int input_index, const double weight);
    void setWeight(const unsigned int input_index, const double weight);


    /** \brief
     *
     * return the bias value for the network
     * \return double
     *
     */
    double getBias() const;
    /** \brief
     * set the bias neuron value
     * \param bias_value const double
     * \return void
     *
     */
    void setBias(const double bias_value);
    double compute(const vector<double> &inputs, bool derivate = false);


    /** \brief
     *   return the weight at index "weight_index" in the neuron.
     * \param weight_index const - index for the weight
     * \return double  - weight value
     *
     */
    double getWeight(const unsigned int weight_index) const;
     /** Stream extraction
    *   \param output
    *   \param neuron reference to print
    *   \return The output stream
    *
    */
    friend ostream & operator<<( ostream &output, const Neuron &neuron)
    {
        output << "Neuron [ index = " << neuron.index << " ]\n" ;
        for(unsigned int i = 0; i < neuron.numberInputs; ++i)
            output << "|" << neuron.w[i] << "|\n";
        output << "|" << neuron.bias << "|\n";
        return output;
    }


};



}
#endif // NEURON_H
