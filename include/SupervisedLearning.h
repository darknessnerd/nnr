#ifndef SUPERVISEDLEARNING_H
#define SUPERVISEDLEARNING_H

#include "Training.h"
#include <string>
using std::string;
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
     *
     *  Add train set from file
     *   with this format input0{delim}input1{delim}...{inputn}{input_target_delim}target0{delim}target1{delim}...{targetm}
     *   for example if we have a neural network that have 3 input value and 2 output value, the file must be like that:
     *   delim:','
     *   input_target_delim:'-'
     *   line_delim:'\n'
     *
     *   3.0,2.0,1.0-2.0,3.0
     *
     *
     * \param file_path const string&
     * \param delimiter char
     * \param line_separator char
     * \return void
     *
     */
    void addTrainSet(const string &file_path, char delim, char line_delim, char input_target_delim);

    /** \brief
     * Add a pair in the training set
     * \param inputs - input of neural network
     * \param targets - expeted input
     * \return void
     *
     */
    void addTrainPair(const vector<double>& inputs, const vector<double>& targets);

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
    vector<string> split(const string &s, char delim);
    vector<double> splitToDouble(const string &s, char delim);


};
}

}


#endif // SUPERVISEDLEARNING_H
