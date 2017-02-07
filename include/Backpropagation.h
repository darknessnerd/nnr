#ifndef BACKPROPAGATION_H
#define BACKPROPAGATION_H

#include "SupervisedLearning.h"
namespace nn
{
namespace train{
class Backpropagation : public SupervisedLearning
{
    public:
        Backpropagation(NeuralNetwork *nn);
        virtual ~Backpropagation();

        void train();

    protected:

    private:
       double learning_rate;
};

}

}

#endif // BACKPROPAGATION_H
