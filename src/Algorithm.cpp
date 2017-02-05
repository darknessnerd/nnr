#include "Algorithm.h"
#include <iostream>
#include <stdexcept>
template <typename T>
double math::algorithm::matrix::det(const math::Matrix<T>& matrix)
{

    std::pair<math::Matrix<double>, math::Matrix<double>> p  = math::algorithm::matrix::crud(matrix);


    double det = 1;
    for(std::size_t i = 0; i < p.first.getRows(); ++i)
    {
       det*= p.first(i,i);
    }

    return det;
}

template <typename T>
std::pair<math::Matrix<double>, math::Matrix<double>> math::algorithm::matrix::crud(const math::Matrix<T>& matrix)
{
    std::size_t n = matrix.getRows();

    if(matrix.getColumns() != matrix.getRows())
       throw std::logic_error("template <typename T> std::pair<math::Matrix<T>, math::Matrix<T>> math::algorithm::matrix::crud(const math::Matrix<T>& matrix) :~ not valid for quadratic matrix");

    std::size_t i, j, k;
    double sum = 0;

    math::Matrix<double> mL(n,n);
    math::Matrix<double> mU(n,n);

    //Upper triandular matrix diagonal
    for(i = 0; i < n; ++i)
    {
       mU(i,i) = 1;
    }


    for (j = 0; j < n; j++)
    {
        for (i = j; i < n; i++)
        {
            sum = 0;
            for (k = 0; k < j; k++)
            {
                sum = sum + mL(i, k) * mU(k, j);
            }
            mL(i, j) = matrix(i, j) - sum;
        }

        for (i = j; i < n; i++)
        {
            sum = 0;
            for(k = 0; k < j; k++)
            {
                sum = sum + mL(j, k) * mU(k, i);
            }
            if (mL(j, j) == 0)
            {
                throw std::logic_error("det(L) close to 0!\n Can't divide by 0...\n");

            }
            mU(j, i) = (matrix(j, i) - sum) /  mL(j, j);
        }
    }

    return std::make_pair(mL, mU);
}
//tempaltes definitsions in this way the compiler can compile this template
template double math::algorithm::matrix::det(const math::Matrix<int>& matrix);
template double math::algorithm::matrix::det(const math::Matrix<double>& matrix);
template double math::algorithm::matrix::det(const math::Matrix<float>& matrix);
template double math::algorithm::matrix::det(const math::Matrix<long>& matrix);
template double math::algorithm::matrix::det(const math::Matrix<unsigned long>& matrix);


template std::pair<math::Matrix<double>, math::Matrix<double>> math::algorithm::matrix::crud(const math::Matrix<double>& matrix);
template std::pair<math::Matrix<double>, math::Matrix<double>> math::algorithm::matrix::crud(const math::Matrix<int>& matrix);
template std::pair<math::Matrix<double>, math::Matrix<double>> math::algorithm::matrix::crud(const math::Matrix<float>& matrix);
template std::pair<math::Matrix<double>, math::Matrix<double>> math::algorithm::matrix::crud(const math::Matrix<long>& matrix);
template std::pair<math::Matrix<double>, math::Matrix<double>> math::algorithm::matrix::crud(const math::Matrix<unsigned long>& matrix);

