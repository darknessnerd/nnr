#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H
#include "Layer.h"
#include <array>
using std::array;
#include <vector>
using std::vector;
namespace nn
{

class NeuralNetwork
{
public:
    NeuralNetwork(const unsigned int numberInputs);
    virtual ~NeuralNetwork();
    unsigned int getNumberInputs() const;
    unsigned int getNumberOutputs() const;
   /** \brief pure virtual void function that is called from the constructor. This function implement the neurla network structure.
     *
     * \return virtual void
     *
     */
    virtual void init() = 0;


    /** \brief
     * Return the matrix weight for the layer with the index "layer_index"
     *
     * \param layer_index const unsignedint
     * \param false bool transpose - transpose the matrix
     * \return Matrix<double> - rows = weight, columns = neuron
     *
     */
    Matrix<double> getWeightMatrixOfLayer(const unsigned int layer_index, bool transpose = false) const;

    /** \brief Compute the layer output
     *
     * \param layer_index const unsignedint - index of the layer to compute
     * \param inputs const vector<double>&  - network input
     * \param false bool derivative=        - use transfer fuction derivative if is set to true
     * \return vector<double>
     *
     */
    vector<double> computeOutputLayer(const unsigned int layer_index, const vector<double> &inputs, bool derivative  = false);
    vector<double> compute(const vector<double> &inputs, bool derivative = false);
       /** \brief
     * \return neuron numbers layer's
     */
    unsigned int getNumLayers() const;

    void random_weight();


    /**
    * \param layer index
    * \param neuron index
    * \param input index
    * \param neuron weight to add
    * \return layer result
    */
    void addWeight(const unsigned int layer_index, const unsigned int neuron_index, const unsigned int input_index, const double weight);

    friend ostream &operator<<( ostream &output, const NeuralNetwork *nn )
    {
        for(unsigned int layer = 0; layer < nn->numLayers; ++layer)
        {
            output << nn->layers[layer];
        }
        return output;
    }
private:

    Layer  **layers = NULL;
    unsigned int numLayers = 0;




protected:
    unsigned int numberInputs;
    unsigned int numberOutputs;

    void addLayer(Layer *layer);
};
}
#endif // NEURALNETWORK_H
