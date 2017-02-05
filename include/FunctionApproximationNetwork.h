#ifndef FUNCTIONAPPROXIMATIONNETWORK_H
#define FUNCTIONAPPROXIMATIONNETWORK_H

#include "NeuralNetwork.h"
class FunctionApproximationNetwork: public nn::NeuralNetwork
{
    public:

        FunctionApproximationNetwork();

        virtual ~FunctionApproximationNetwork();
        void init();
    protected:

    private:
};



#endif // FUNCTIONAPPROXIMATIONNETWORK_H
