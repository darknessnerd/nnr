#ifndef TRANSFERFUNCTION_H
#define TRANSFERFUNCTION_H
#include<exception>
#include<math.h>
namespace nn
{

class TransferFunctionTypeException: public std::exception
{
  virtual const char* what() const throw()
  {
    return "Not valid transfer function exception!";
  }
};
enum class TransferFunctionType
{
    HardLimitType, HardLimitSymmetricalType, PureLinearType, LogSigType
};
class TransferFunction
{
public:
    /** Default constructor */
    TransferFunction();
    virtual double compute(double n) = 0;
    virtual double derivative(double n) = 0;
    /** Default destructor */
    virtual ~TransferFunction();
protected:
private:
};


class HardLimit: public TransferFunction
{
    double compute(double n)
    {
        if(n < 0)
            return 0;
        return 1;
    }
    double derivative(double n)
    {
        return 0;
    }
};
class LogSigmoid: public TransferFunction
{
    double compute(double n)
    {
        return 1/(1+exp(-1*n));
    }
    double derivative(double n)
    {
        return (1-n)*n;
    }
};
class HardLimitSymmetrical: public TransferFunction
{
    double compute(double n)
    {
        if(n < 0)
            return -1;
        return 1;
    }
    double derivative(double n)
    {
        return 0;
    }
};

class PureLinear: public TransferFunction
{
    double compute(double n)
    {
        return n;
    }
    double derivative(double n)
    {
        return 1;
    }
};
}
#endif // TRANSFERFUNCTION_H
