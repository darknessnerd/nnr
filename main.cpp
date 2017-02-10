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

vector<string> split(const string &s, char delim)
{
    stringstream ss(s);
    string item;
    vector<string> tokens;
    while (getline(ss, item, delim))
    {
        tokens.push_back(item);
    }
    return tokens;
}
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



    cout << endl;
    nn::train::Backpropagation bck(fan);
    //add training elemnts


    std::ifstream file("train.csv");
    string line;
    if (file.is_open())
    {
        while ( getline (file,line, '\n') )
        {
            cout << line << '\n';
            vector<string> parsed = split(line, ',');
            double *d = new double[parsed.size()];
            int i = 0;
            for(std::vector<string>::const_iterator output_iter = parsed.begin(); output_iter != parsed.end(); ++output_iter)
            {
                std::string::size_type sz;     // alias of size_t

                double value = std::stod (*output_iter,&sz);
                d[i++] = value;
            }
            bck.addTrainPair({d[0]}, {d[1]});

            delete d;

        }

        file.close();
    }
    else cout << "Unable to open file";


    bck.train();


    cout << "after train: \n";
    cout << fan << endl;


    std::ifstream file1("input.csv");
    if (file1.is_open())
    {
        while ( getline (file,line, '\n') )
        {
            cout << line << '\n';

            double value = std::stod (line);

            vector<double> r = fan->compute({value});

        }

        file1.close();
    }
    else cout << "Unable to open file";



    delete fan;

}

int main()
{



    backPropagationTest();

    return 0;
}





