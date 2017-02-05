#ifndef SUPERVISEDLEARNING_H
#define SUPERVISEDLEARNING_H

#include "Training.h"

#include <list>
#include <utility>
using std::pair;
using std::list;
namespace nn
{
namespace train
{
class SupervisedLearning : public Training
{
public:
    /** Default constructor */
    SupervisedLearning( NeuralNetwork* const nn);
    /** Default destructor */
    virtual ~SupervisedLearning();
    /** \brief
     * Add a pair in the training set
     * \param inputs - input of neural network
     * \param targets - expeted input
     * \return void
     *
     */
    void addTrainPair(const list<double>& inputs, const list<double>& targets);

    virtual void train() = 0;

     /** Stream extraction
    *   \param output
    *   \param neuron reference to print
    *   \return The output stream
    *
    */
    friend ostream& operator<<( ostream &output, const SupervisedLearning& t)
    {
       output << "[ Training set ]\n" ;
       for(std::list<pair<double*, double*>>::iterator it = t.testset->begin(); it != t.testset->end(); ++it)
       {
           double *first =  (*it).first;
           double *second = (*it).second;
           output << "[ ";
           for(unsigned int i = 0; i < t.nn->getNumberInputs(); ++i)
           {
                output << first[i] <<  " " ;
           }
           output << "| ";
           for(unsigned int i = 0; i < t.nn->getNumberOutputs(); ++i)
           {
                output << second[i] <<  " " ;
           }

           output << "]\n";
       }

       return output;
    }
protected:
    list<pair<double*, double*>> *testset;
private:

};
}

}


#endif // SUPERVISEDLEARNING_H
