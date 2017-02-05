#include "HubeRuleSupervised.h"
#include "Matrix.h"
using namespace nn::train;
using namespace math;

HubeRuleSupervised::HubeRuleSupervised(NeuralNetwork* const nn):SupervisedLearning(nn)
{
}
void HubeRuleSupervised::train()
{
    //HubeRuleSupervised
    //T is the matrix where rows = number outputs columns = number test case
    //P is the matrix where rows = number inputs columns = number testa case
    // W = TP^t
    /**
    * W matrix rows = number neurons in the layer, columns = numbers outputs of the layer
    **/
    std::size_t number_test_pair = this->testset->size();


    // TODO (darknessnerd#1#): check hube if is column independe and implement the pseudo inverse
    if(nn->getNumberInputs() > number_test_pair )
      throw std::logic_error("to implement");

    Matrix<double> T(nn->getNumberOutputs(), number_test_pair);
    Matrix<double> Ptras(number_test_pair, nn->getNumberInputs());

    //construct matrix T and P
    std::size_t test_pair_count = 0;
    for(std::list<pair<double*,double*>>::iterator n = testset->begin(); n != testset->end(); ++n)
    {
        //input - target
        double* input  = (*n).first;
        double* target = (*n).second;


        for(std::size_t i = 0; i< nn->getNumberOutputs(); ++i)
           T(i, test_pair_count) = target[i];

        for(std::size_t i = 0; i< nn->getNumberInputs(); ++i)
           Ptras(test_pair_count, i) = input[i];



        test_pair_count++;
    }

    Matrix<double> W = T*Ptras;
    for(std::size_t neuron_index = 0; neuron_index< nn->getNumberOutputs(); ++neuron_index)
       for(std::size_t input_index = 0; input_index< nn->getNumberInputs(); ++input_index)
       {
          this->nn->addWeight(0,neuron_index,input_index,W(neuron_index, input_index));
       }
    std::cout << W << std::endl;

}

HubeRuleSupervised::~HubeRuleSupervised()
{
    //dtor
}
