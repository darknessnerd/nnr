#ifndef ALGORITHM_H
#define ALGORITHM_H

#include "Matrix.h"

#include <utility>
namespace math

{

    namespace algorithm
    {
        namespace matrix
        {
            template <typename T>
            double det(const math::Matrix<T>& matrix);


            /**! \brief Crout matrix decomposition A = LU
            *
            * \param matrixA - matrix to decomposes
            * \return pair<L,U> matrixs
            **/
            template <typename T>
            std::pair<math::Matrix<double> ,math::Matrix<double>> crud(const math::Matrix<T>& matrix);




        }
    }

}

#endif // ALGORITHM_H
