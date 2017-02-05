#ifndef PERCEPTRON_H
#define PERCEPTRON_H
#include "NeuralNetwork.h"
/**
   Perceptron
   The perceptron can be used to classify input vectors that can be separated by a linear boundary. We call such
    vectors linearly separable.
**/
class Perceptron: public nn::NeuralNetwork
{
private:
   unsigned int numNeurons;
public:
    Perceptron(const unsigned int &numInputs, const unsigned int &numNeurons);
    ~Perceptron();
    void init();
};


#endif // PERCEPTRON_H
