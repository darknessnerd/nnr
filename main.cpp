#include <iostream>
using namespace std;
#include "NeuralNetwork.h"
#include "PerceptronLearningRule.h"
#include "HubeRuleSupervised.h"
#include "Example.h"
#include <typeinfo>
#include <array>
#include <utility>
#include "Algorithm.h"
#include "FunctionApproximationNetwork.h"
#include "Backpropagation.h"

#include "Perceptron.h"

#include <iostream>
#include <fstream>
#include <sstream>


template <typename T> void printcoll (T const& coll)
{
    typename T::const_iterator pos;  // iterator to iterate over coll
    typename T::const_iterator end(coll.end());  // end position
    std::cout << "--- ";
    for (pos=coll.begin(); pos!=end; ++pos)
    {
        std::cout << *pos << ' ';

    }

    std::cout << "--- ";
    std::cout << std::endl;
}

void hubeRuleTest()
{
    nn::NeuralNetwork *perceptron = new Perceptron(2,1);
    perceptron->init();



    cout << endl;
    nn::train::HubeRuleSupervised plr(perceptron);
    //add training elemnts
    plr.addTrainPair({1,2}, {1});
    plr.addTrainPair({-1,2}, {-1});
    plr.addTrainPair({0,-1}, {-1});

    plr.train();

    cout << perceptron << endl;


    vector<double> r = perceptron->compute({3,1});
    printcoll(r);
    r = perceptron->compute({3.5,0.5});
    printcoll(r);
    r = perceptron->compute({-1,1});
    printcoll(r);

    delete perceptron;

}

void perceptronTest()
{

    nn::NeuralNetwork *perceptron = new Perceptron(2,1);
    perceptron->init();



    cout << endl;
    nn::train::PerceptronLearningRule plr(perceptron);
    //add training elemnts
    plr.addTrainPair({1,2}, {1});
    plr.addTrainPair({-1,2}, {0});
    plr.addTrainPair({0,-1}, {0});

    plr.train();

    cout << perceptron << endl;


    vector<double> r = perceptron->compute({3,1});
    printcoll(r);
    r = perceptron->compute({3.5,0.5});
    printcoll(r);
    r = perceptron->compute({-1,1});
    printcoll(r);

    delete perceptron;
}

void backPropagationTest()
{

    nn::NeuralNetwork *fan = new FunctionApproximationNetwork();
    fan->init();
    nn::train::Backpropagation bck(fan, 0.1,1);
    //add training elemnts from file
    bck.addTrainSet("train.csv", ',', '\n', '|');
    bck.train();
    fan->compute("input.csv", "result.dat", ',','\n',',');
    delete fan;

}

int main()
{
    backPropagationTest();

    return 0;
}





