#include "Matrix.h"
using namespace math;
#include <utility>
#include <stdexcept>
#include <iostream>
using std::cout;



template<class T>
Matrix<T>::Matrix(const std::size_t rows, const std::size_t columns):rows(rows), columns(columns)
{
    std::cout << "template<class T> Matrix<T>::Matrix(const std::size_t rows, const std::size_t columns):rows(rows), columns(columns)\n";
    this->m = new T*[rows];
    for( std::size_t i = 0; i < rows; ++i)
    {
        this->m[i] = new T[columns];
        for( std::size_t j = 0; j < columns; ++j)
            this->m[i][j] = 0;

    }
}


template<class T>
Matrix<T>::Matrix(const std::initializer_list<std::initializer_list<T>> matrix):m(nullptr)
{
     std::cout << "template<class T> Matrix<T>::Matrix(const std::initializer_list<std::initializer_list<T>>& matrix):m(nullptr)\n";
    if(matrix.size() > 0)
    {
        this->rows = matrix.size();
        this->m = new T*[rows];

        typename std::initializer_list<std::initializer_list<T>>::iterator it;  // same as: const int* it
        unsigned int row_index = 0;
        this->columns = 0;
        for ( it=matrix.begin(); it!=matrix.end(); ++it)
        {

            std::initializer_list<T> columns_list = *it;
            if(this->columns != 0 && this->columns != columns_list.size())
                throw std::length_error("matrix columns size different!");
            this->columns = columns_list.size();
            this->m[row_index] = new T[this->columns];
            unsigned int col_index = 0;
            for ( typename  std::initializer_list<T>::iterator col=columns_list.begin(); col!=columns_list.end(); ++col)
               this->m[row_index][col_index++] = *col;
            row_index++;
        }

    }
}
template<class T>
Matrix<T>::Matrix(const Matrix<T> &other)
{
    cout << "template<class T> Matrix<T>::Matrix(const Matrix<T> &other)\n";
    this->copyFrom(other);
}
template<class T>
Matrix<T>::Matrix(Matrix<T> &&other) noexcept
{
    cout << "template<class T> Matrix<T>::Matrix(Matrix<T> &&other) noexcept\n";
    //COPY FROM OTHER
    this->copyFrom(other);
    //RESET THE OTHER OBJECT
    if(other.rows)
    {
        if(other.columns)
        {
            for( std::size_t  i = 0; i < columns; ++i)
                delete [] other.m[i];
        }
        delete [] other.m;
    }
    other.rows = 0;
    other.columns = 0;

}
template<class T>
void Matrix<T>::copyFrom(const Matrix<T> &other)
{
    cout << "template<class T> void Matrix<T>::copyFrom(const Matrix<T> &other)\n";
    this->rows = other.rows;
    this->columns = other.columns;

    if( this->rows)
    {
        this->m = new T*[rows];
        for( std::size_t  i = 0; i < rows; ++i)
        {
            this->m[i] = nullptr;
            if(columns)
            {
                this->m[i] = new T[columns];
                for( std::size_t j = 0; j < columns; ++j)
                    this->m[i][j] = other.m[i][j];
            }
        }
    }
    else
    {
        this->m = nullptr;
    }

}
/** Copy assignment operator */
template<class T>
Matrix<T>& Matrix<T>::operator= (const Matrix<T>& other)
{
    std::cout << "template<class T> Matrix<T>& Matrix<T>::operator= (const Matrix<T>& other)\n";
    Matrix<T> tmp(other);         // re-use copy-constructor
    *this = std::move(tmp); // re-use move-assignment
    return *this;
}

/** Move assignment operator */
template<class T>
Matrix<T>& Matrix<T>::operator= (Matrix<T>&& other) noexcept
{
    std::cout << "template<class T> Matrix<T>& Matrix<T>::operator= (Matrix<T>&& other) noexcept\n";
    //delete current data
    for( std::size_t i = 0; i < rows; ++i)
        delete []this->m[i];
    delete [] this->m;
    //move all data
    this->rows = other.rows;
    this->columns = other.columns;
    this->m = other.m;
    //clean the other object
    other.m = nullptr;
    other.rows =  other.columns = 0;
    return *this;
}

template<class T>
Matrix<T>::Matrix():rows(0), columns(0)
{
    this->m = nullptr;
}

template<class T>
Matrix<T>::~Matrix()  noexcept
{
    std::cout << "template<class T> Matrix<T>::~Matrix()  noexcept\n";
    for( std::size_t i = 0; i < rows; ++i)
        delete []this->m[i];

    delete [] this->m;
}
template<class T> T& Matrix<T>::operator()(std::size_t row, std::size_t column)
{
    return this->m[row][column];
}
template<class T> T Matrix<T>::operator()(std::size_t row, std::size_t column) const
{
    return this->m[row][column];
}

template<class T> Matrix<T> Matrix<T>::operator*(const Matrix<T> & other)
{
   if(this->columns != other.rows)
      throw std::logic_error("template<class T> Matrix<T>&& Matrix<T>::operator()(const Matrix<T> other) :~ invalid multiplication matrix  ");
   Matrix<T> result(this->rows, other.columns);

   for(std::size_t cOther = 0; cOther < other.columns; ++cOther)
   {
      for(std::size_t row = 0; row < this->rows; ++row)
      {
         for(std::size_t multiplicationIndex = 0; multiplicationIndex < this->columns; ++multiplicationIndex)
         {
             T n1 = this->m[row][multiplicationIndex];
             T n2 = other.m[multiplicationIndex][cOther];

             result(row, cOther) += n1*n2;
         }

      }
   }

   return result;

}
template<class T> Matrix<T> Matrix<T>::operator*(double constant)
{
   Matrix<T> result(this->rows, this->columns);

   for(std::size_t row = 0; row < this->rows; ++row)
   {
      for(std::size_t column = 0; column < this->columns; ++column)
      {
          result(row, column) = this->m[row][column]*constant;

      }
   }
   return result;

}
template<class T>
 std::size_t Matrix<T>::getRows() const
 {
    return this->rows;
 }
 template<class T>
std::size_t Matrix<T>::getColumns()  const
{
   return this->columns;
}
template class Matrix<float>;
template class Matrix<bool>;
template class Matrix<double>;
template class Matrix<int>;
template class Matrix<long>;
template class Matrix<unsigned long>;



