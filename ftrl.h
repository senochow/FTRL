/* ############################################################################
* 
* Copyright (c) 2016 ICT MCG Group, Inc. All Rights Reserved
* 
* ###########################################################################
* Brief: 
 
* Authors: zhouxing(@ict.ac.cn)
* Date:    2016/11/23 09:29:17
* File:    ftrl.h
*/
#include <iostream>
using namespace std;


class FTRL {
    private:
        double alpha;
        double beta;
        double L1;
        double L2;

        int dim;
        // squared sum of past gradients
        vector<double> n;
        // weights z
        vector<double> z;
        // model weights
        vector<double> w;
    public:
        FTRL(double _alpha, double _beta, double _L1, double _L2, int _dim);
        double predict(vector<int> x);
        double update(vector<int> x, double p, int y);
};
