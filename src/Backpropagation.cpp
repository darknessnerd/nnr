#include "Backpropagation.h"
using namespace nn::train;

#include <math.h>
#include "Matrix.h"
using namespace math;
Backpropagation::Backpropagation(NeuralNetwork *nn):SupervisedLearning(nn)
{

}

Backpropagation::~Backpropagation()
{

}
void Backpropagation::train()
{
    //random weight
    nn->random_weight();

    unsigned int numInputs = nn->getNumberInputs();
    unsigned int numLayers = nn->getNumLayers();
    std::cout << numLayers << " ################ \n";
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
        for(std::vector<double>::const_iterator n = a.begin(); n != a.end(); ++n)
        {
            double e = target[output_index]-(*n);
            //
            error[output_index] =   0.5*pow(e, 2.0) ;
            output_index++;
            total_error+=error[output_index];
        }
        //THIRD STEP BACKPROPAGATION
        //STEP S^m where m last layer

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
        Matrix<double> sMResult = fDerivativeMatrix*errorMatrix*-2;


        //S-layer_number

        for(int  l = numLayers-2; l >= 0 ; --l)
        {
            std::cout << " layer number : "  << l << "\n";
            Matrix<double> fDerM();


            fDerivativeM = nn->computeOutputLayer(l,inputVector, true);

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


        }


        delete error;
    }
}
