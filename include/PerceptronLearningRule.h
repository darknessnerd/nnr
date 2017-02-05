#ifndef PERCEPTRONLEARNINGRULE_H
#define PERCEPTRONLEARNINGRULE_H

#include "SupervisedLearning.h"

namespace nn
{
namespace train
{
class PerceptronLearningRule : public SupervisedLearning
{
public:
    /** Default constructor */
    PerceptronLearningRule(NeuralNetwork* const nn);
    /** Default destructor */
    virtual ~PerceptronLearningRule();

    void train();

protected:

private:
};
}
}


#endif // PERCEPTRONLEARNINGRULE_H
