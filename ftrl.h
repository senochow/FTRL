
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
#ifndef FTRL_H_
#define FTRL_H_

#include <iostream>
#include <unordered_map>
#include <vector>
#include <cmath>

using namespace std;


class FTRL {
    private:
        // learning rate
        double alpha;
        // smoothing value
        double beta;
        double L1;
        double L2;

        int dim;
        // squared sum of past gradients
        vector<double> n;
        // weights z
        vector<double> z;
        // model lazy weights
        // vector<double> w;
        unordered_map<int, double> w;
    public:
        FTRL(double _alpha, double _beta, double _L1, double _L2, int _dim);
        double predict(vector<int> x);
        double update(vector<int> x, double p, int y);
};

FTRL::FTRL(double _alpha, double _beta, double _L1, double _L2, int _dim) {
    alpha = _alpha;
    beta = _beta;
    L1 = _L1;
    L2 = _L2;
    dim = _dim;
    n = vector<double>(dim, 0);
    z = vector<double>(dim, 0);
}

double FTRL::predict(vector<int> x) {
    double wTx = 0;
    unordered_map<int, double> _w;
    for (size_t i = 0; i < x.size(); i++) {
        // update weight w in prediction stage, lazy weights
        int sign = (z[x[i]] < 0) ? -1: 1;
        if (sign*z[x[i]] <= L1) {
            _w[x[i]] = 0;
        } else {
            _w[x[i]] = (sign*L1 - z[x[i]]) / ((beta + sqrt(n[x[i]]))/alpha + L2);
        }
        wTx += _w[x[i]];
    }
    // cache current w for update stage
    w = _w;
    // sigmoid function
    return 1.0/(1.0 + exp(-wTx));
}



#endif  // FTRL_H_
