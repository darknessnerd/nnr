#include "NeuralNetworkUtil.h"


#include <iostream>
#include <fstream>
#include <sstream>


vector<string> nn::util::split_str(const string &s, char delim)
        {
            vector<string> tokens;
            std::stringstream ss(s);
            string item;

            while (getline(ss, item, delim))
                tokens.push_back(item);

            return tokens;
        }
vector<double> nn::util::split_double(const string &s, char delim)
        {
            vector<double> tokens;
            std::stringstream ss(s);
            string item;

            while (getline(ss, item, delim))
                tokens.push_back(std::stod(item));

            return tokens;
        }
