#include "SupervisedLearning.h"
using namespace nn::train;

SupervisedLearning::SupervisedLearning( NeuralNetwork* const nn):Training(nn)
{
    this->testset = new list<pair<double*, double*>>();
}
void SupervisedLearning::addTrainPair(const list<double>& inputs, const list<double>& targets)
{

    if(inputs.size() != this->nn->getNumberInputs())
        throw std::invalid_argument("void SupervisedLearning::addTrainPair(const list<double>& inputs, const list<double>& targets) :~ Invalid size input array");
    if(targets.size() != this->nn->getNumberOutputs())
        throw std::invalid_argument("void SupervisedLearning::addTrainPair(const list<double>& inputs, const list<double>& targets) :~ Invalid size targets array");

    double *i = new double[this->nn->getNumberInputs()];
    double *o = new double[this->nn->getNumberOutputs()];

    unsigned int count = 0;
    for(std::list<double>::const_iterator n = inputs.begin(); n != inputs.end(); ++n)
       i[count++] = (*n);
    count = 0;
    for(std::list<double>::const_iterator n = targets.begin(); n != targets.end(); ++n)
       o[count++] = (*n);


    pair<double*,double*> p = std::make_pair(i,o);
    this->testset->push_back(p);
}
SupervisedLearning::~SupervisedLearning()
{
    std::cout << "~SupervisedLearning";
    for(std::list<pair<double*,double*>>::iterator n = testset->begin(); n != testset->end(); ++n)
    {
        delete (*n).first;
        delete (*n).second;
    }
    delete this->testset;
}
