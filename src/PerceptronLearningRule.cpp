#include "PerceptronLearningRule.h"
using namespace nn::train;
PerceptronLearningRule::PerceptronLearningRule(NeuralNetwork* const nn):SupervisedLearning(nn)
{
    //ctor
}

PerceptronLearningRule::~PerceptronLearningRule()
{
    std::cout << "~PerceptronLearningRule" << std::endl;
}


void PerceptronLearningRule::train()
{
    unsigned int inputSize = nn->getNumberInputs();
    //random weight
    for(unsigned int input_index = 0 ; input_index <  inputSize; ++input_index)
    {
        for(unsigned int neuron_index = 0 ;  neuron_index < nn->getNumberOutputs(); ++neuron_index)
        {

            this->nn->addWeight(0,neuron_index,input_index,0.5);

        }
    }
    bool completed = false;
    while(!completed)
    {

        for(std::list<pair<double*,double*>>::iterator n = testset->begin(); n != testset->end(); ++n)
        {
            //input - target
            double* input  = (*n).first;
            double* target = (*n).second;

            vector<double> inputVector = vector<double>(input,(input+inputSize));
            //a - value classified from the current network
            vector<double> a = nn->compute(inputVector);

            //compute te error from the tagert vector
            double *error = new double[nn->getNumberOutputs()];
            unsigned int neuron_index = 0;
            for(std::vector<double>::const_iterator n = a.begin(); n != a.end(); ++n)
            {
                error[neuron_index] =   target[neuron_index]-(*n) ;
                cout << (*n) << "-"  << target[neuron_index]  << " =  " <<  error[neuron_index]<< "\n" ;
                neuron_index++;
            }
            completed = true;
            //update the errors
            for(unsigned int input_index = 0 ; input_index <  inputSize; ++input_index)
            {
                for(unsigned int neuron_index = 0 ;  neuron_index < nn->getNumberOutputs(); ++neuron_index)
                {
                    if(error[neuron_index] != 0)
                    {
                        double value = error[neuron_index]*input[input_index];
                        this->nn->addWeight(0,neuron_index,input_index,value);
                        completed = false;
                    }
                }
            }


            //Gemerate random weight tiny
            delete error;
        }
    }


}
