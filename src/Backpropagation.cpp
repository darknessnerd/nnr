#include "Backpropagation.h"
using namespace nn::train;

#include <math.h>
#include "Matrix.h"
using namespace math;
Backpropagation::Backpropagation(NeuralNetwork *network, double learning_rate, double momentum)
:SupervisedLearning(network),
 learning_rate(learning_rate),
  momentum(momentum)
{

}

Backpropagation::~Backpropagation()
{

}
void Backpropagation::train()
{
    //random weight
    nn->random_weight();



    unsigned int epoch = 0;
    while(epoch < 350)
    {
        epoch++;
        unsigned int numInputs = nn->getNumberInputs();
        unsigned int numLayers = nn->getNumLayers();
        for(std::list<pair<double*,double*>>::iterator n = testset->begin(); n != testset->end(); ++n)
        {
            //input - target
            double* input  = (*n).first;
            double* target = (*n).second;
            //FIRST STEP COMPUTE THE OUTPUT FORM THE NN
            vector<double> inputVector = vector<double>(input,(input+numInputs));
            //a is the vector output of the nn
            vector<double> a = nn->compute(inputVector);
            //SECOND STEP COMPUTE THE ERROR
            double *error = new double[nn->getNumberOutputs()];
            double total_error = 0;
            unsigned int output_index = 0;
            for(std::vector<double>::const_iterator output_iter = a.begin(); output_iter != a.end(); ++output_iter)
            {
                double e = target[output_index]-(*output_iter);
                //
                error[output_index] =   e;//0.5*pow(e, 2.0) ;
                output_index++;
                total_error+=error[output_index];
            }
            //THIRD STEP BACKPROPAGATION
            //STEP S^m where m last layer
            Matrix<double> *sensitivies = new Matrix<double>[numLayers];

            //INIT ALL ELEMENT
            vector<double> fDerivativeM = nn->compute(inputVector, true);
            Matrix<double> errorMatrix(nn->getNumberOutputs(), 1);
            for(size_t i = 0; i < nn->getNumberOutputs(); ++i)
                errorMatrix(i,0) = error[i];
            Matrix<double> fDerivativeMatrix(fDerivativeM.size(), fDerivativeM.size());
            for(size_t i = 0; i < fDerivativeM.size(); ++i)
            {
                for(size_t j = 0; j < fDerivativeM.size(); ++j)
                {
                    if(i==j)
                        fDerivativeMatrix(i,j) = fDerivativeM[i];
                    else
                        fDerivativeMatrix(i, j) = 0.0;
                }
            }
            sensitivies[numLayers-1] = fDerivativeMatrix*errorMatrix*-2;

            //S-layer_number
            for(int  layer_index = numLayers-2; layer_index >= 0 ; --layer_index)
            {
                //compute matrix for F^layer_index(n^layer_index)
                Matrix<double> fDerM();
                fDerivativeM = nn->computeOutputLayer(layer_index,inputVector, true);
                fDerivativeMatrix =  Matrix<double>(fDerivativeM.size(), fDerivativeM.size());
                for(size_t i = 0; i < fDerivativeM.size(); ++i)
                {
                    for(size_t j = 0; j < fDerivativeM.size(); ++j)
                    {
                        if(i==j)
                            fDerivativeMatrix(i,j) = fDerivativeM[i];
                        else
                            fDerivativeMatrix(i, j) = 0.0;
                    }
                }
                //Weight transpose matrix layer_index+1
                Matrix<double> wT = nn->getWeightMatrixOfLayer(layer_index+1, true);
                sensitivies[layer_index] = fDerivativeMatrix*wT*sensitivies[layer_index+1];

            }
            //LAST STEP - update the weights
            for(int  layer_index = numLayers-1; layer_index >= 0 ; --layer_index)
            {

                Matrix<double> w = nn->getWeightMatrixOfLayer(layer_index);
                //compute the input for the current layer

                vector<double> network_output;
                if(layer_index != 0)
                    network_output = nn->computeOutputLayer(layer_index-1,inputVector, false);
                else
                    network_output = inputVector;
                Matrix<double> aT(1, network_output.size());
                for(unsigned int i = 0; i < network_output.size(); ++i)
                    aT(0,i) = network_output[i];

                Matrix<double> wToUpdate = w-(sensitivies[layer_index] * aT * learning_rate);
                this->nn->setWeights(layer_index, wToUpdate);

                Matrix<double> biasToUpdate = nn->getBiasMatrixOfLayer(layer_index) - ( sensitivies[layer_index] * learning_rate );
                this->nn->setBiases(layer_index, biasToUpdate);

            }

            delete []sensitivies;
            delete error;
        }
    }
}
