#ifndef AUTOMATACELLULAR_H
#define AUTOMATACELLULAR_H

#include "Matrix.h"
namespace ac {

class AutomataCellular
{
    public:
        /** Default constructor */
        AutomataCellular();
        /** Default destructor */
        virtual ~AutomataCellular();

    protected:


    private:

       unsigned long int current_step;

       math::Matrix<double> *currents;
       math::Matrix<double> *nexts;
};

}
#endif // AUTOMATACELLULAR_H
