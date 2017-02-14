#include "SupervisedLearning.h"
using namespace nn::train;

#include "NeuralNetworkUtil.h"

#include <iostream>
#include <fstream>
#include <sstream>

SupervisedLearning::SupervisedLearning( NeuralNetwork* const nn):Training(nn)
{
    this->testset = new list<pair<double*, double*>>();
}


void SupervisedLearning::addTrainSet(const string &file_path, char delim, char line_delim, char input_target_delim)
{
    std::ifstream file(file_path);
    string line;
    if (file.is_open())
    {
        while ( getline (file,line, line_delim) )
        {
            vector<string> tuple2 = nn::util::split_str(line, input_target_delim);

            if(tuple2.size() != 2)
                throw std::runtime_error("SupervisedLearning::addTrainSet - invalid file input format! ");

            //parse the string number0{delim}number1{delim}...{delim}numbern
            vector<double> input = nn::util::split_double(tuple2[0], delim);
            //parse the string number0{delim}number1{delim}...{delim}numbern
            vector<double> target = nn::util::split_double(tuple2[1], delim);


            if(input.size() != nn->getNumberInputs())
                throw std::runtime_error("SupervisedLearning::addTrainSet - invalid file input format!");
            if(target.size() != nn->getNumberOutputs())
                throw std::runtime_error("SupervisedLearning::addTrainSet - invalid file input format!");

            this->addTrainPair(input, target);


        }

        file.close();
    }
    else cout << "Unable to open file";
}

void SupervisedLearning::addTrainPair(const vector<double>& inputs, const vector<double>& targets)
{

    if(inputs.size() != this->nn->getNumberInputs())
        throw std::invalid_argument("void SupervisedLearning::addTrainPair(const list<double>& inputs, const list<double>& targets) :~ Invalid size input array");
    if(targets.size() != this->nn->getNumberOutputs())
        throw std::invalid_argument("void SupervisedLearning::addTrainPair(const list<double>& inputs, const list<double>& targets) :~ Invalid size targets array");

    double *i = new double[this->nn->getNumberInputs()];
    double *o = new double[this->nn->getNumberOutputs()];

    unsigned int count = 0;
    for(std::vector<double>::const_iterator n = inputs.begin(); n != inputs.end(); ++n)
       i[count++] = (*n);
    count = 0;
    for(std::vector<double>::const_iterator n = targets.begin(); n != targets.end(); ++n)
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
