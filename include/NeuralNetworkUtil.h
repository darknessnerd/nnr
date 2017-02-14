#ifndef NEURALNETWORKUTIL_H
#define NEURALNETWORKUTIL_H

#include<vector>
using std::vector;
#include<string>
using std::string;

namespace nn
{
    namespace util
    {

        vector<string> split_str(const string &s, char delim);
        vector<double> split_double(const string &s, char delim);
    }
}

#endif // NEURALNETWORKUTIL_H
