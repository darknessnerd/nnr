#ifndef BACKPROPAGATION_H
#define BACKPROPAGATION_H

#include "SupervisedLearning.h"
namespace nn
{
namespace train{
class Backpropagation : public SupervisedLearning
{
    public:

        /** \brief
         *
         * \param network NeuralNetwork*
         * \param
         * \param learning_rate double
         * \param 1.0 double momentum= is a Heuristic Modifications of Backpropagation, 0 <= momentum <= 1
         *
         */
        Backpropagation(NeuralNetwork *network, double learning_rate, double momentum = 1.0);
        virtual ~Backpropagation();

        void train();

    protected:

    private:
       double learning_rate;
       double momentum;
};

}

}

#endif // BACKPROPAGATION_H
