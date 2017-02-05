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

    void addWeight(const unsigned int &input_index, const double &weight);




    double compute(const vector<double> &inputs, bool derivate = false);

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
