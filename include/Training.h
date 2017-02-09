#ifndef TRAINING_H
#define TRAINING_H
#include "NeuralNetwork.h"
namespace nn
{
    namespace train
    {

    class Training
    {

        public:
            /** Default constructor */
            Training( NeuralNetwork* const network);
            /** Default destructor */
            virtual ~Training() = 0;

        protected:
            NeuralNetwork* const nn;
        private:
    };
    }
}
#endif // TRAINING_H
