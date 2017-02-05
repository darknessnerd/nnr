#ifndef MATRIX_H
#define MATRIX_H
#include <iostream>
#include <initializer_list>  // std::initializer_list
namespace math
{


template<class T>
class Matrix
{
public:
    Matrix(const std::size_t rows, const std::size_t columns);
    Matrix(const std::initializer_list<std::initializer_list<T>> list);
    /** Default constructor */
    Matrix();
    /** Copy Construcotr **/
    Matrix(const Matrix<T> & other);
     /** Move constructor */
    Matrix(Matrix<T>&& other) noexcept; /* noexcept needed to enable optimizations in containers */
     /** Copy assignment operator */
    Matrix& operator= (const Matrix<T>& other);

    /** Move assignment operator */
    Matrix& operator= (Matrix<T>&& other) noexcept;

    /** Default destructor */
    virtual ~Matrix() noexcept;

    /** \brief
     *
     * Return the object in the matrix in position [row][column]
     *
     *
     * \param index row
     * \param index column
     * \return
     *
     */
    T& operator()(std::size_t row, std::size_t column);

    /** \brief
     *
     * Return the object in the matrix in position [row][column]
     *
     *
     * \param index row
     * \param index column
     * \return
     *
     */
    T operator()(std::size_t row, std::size_t column) const;

    /** \brief
     *
     * \param Matrix<T> other
     * \return Return Matrix result from  the operation this x other
     *
     */
    Matrix<T> operator*(const Matrix<T> & other);
    Matrix<T> operator*(double constant);

    std::size_t getRows() const;
    std::size_t getColumns() const;
protected:

private:
    T **m;

    std::size_t rows;
    std::size_t columns;

    void copyFrom(const Matrix<T> & other);

    friend std::ostream& operator<< (std::ostream& os, const Matrix<T>& m)
    {

        for(std::size_t i = 0; i < m.rows; ++i)
        {
           os << "| ";
           for(std::size_t j = 0; j < m.columns; ++j)
               os << m.m[i][j] << " ";
           os << "|" << std::endl;
        }
        return os;
    }
};
}
#endif // MATRIX_H
