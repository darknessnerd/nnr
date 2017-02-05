#ifndef HUBERULESUPERVISED_H
#define HUBERULESUPERVISED_H
#include "SupervisedLearning.h"
namespace nn
{
namespace train
{

class HubeRuleSupervised : public SupervisedLearning
{
    public:
        /** Default constructor */
        HubeRuleSupervised(NeuralNetwork* const nn);
        /** Default destructor */
        virtual ~HubeRuleSupervised();

        void train();

    protected:

    private:
};
}
}
#endif // HUBERULESUPERVISED_H
