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


#include "Perceptron.h"
// print elements of an STL container
template <typename T> void printcoll (T const& coll);

void perceptronTest();
void hubeRuleTest();


int main()
{

    math::Matrix<double> m(
    {
        {1, 2, 4},
        {3, 8 , 14},
        {2, 6 ,13}
    }
    );

    std::pair<math::Matrix<double>, math::Matrix<double>> r = math::algorithm::matrix::crud(m);
    double det = math::algorithm::matrix::det(m);
    std::cout << r.first;
    std::cout << std::endl;
    std::cout << r.second;
    std::cout << std::endl;
    std::cout << r.first * r.second;

    std::cout << "determinant: " << det << std::endl;

    //hubeRuleTest();
    //perceptronTest();

    return 0;
}
template <typename T>
void printcoll (T const& coll)
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





