
/* ############################################################################
* 
* Copyright (c) 2016 ICT MCG Group, Inc. All Rights Reserved
* 
* ###########################################################################
* Brief: 
  store model weights w  
  more faster than lazy mode
* Authors: zhouxing(@ict.ac.cn)
* Date:    2016/11/23 09:29:17
* File:    ftrl_v1.h
*/
#ifndef FTRL_H_
#define FTRL_H_

#include <iostream>
#include <unordered_map>
#include <vector>
#include "util.h"
using namespace std;


class FTRL {
    private:
        // learning rate
        double alpha;
        // smoothing value
        double beta;
        double L1;
        double L2;

        // squared sum of past gradients
        vector<double> n;
        // weights z
        vector<double> z;
        // model weights
        vector<double> w;
    public:
        int dim;
        FTRL(double _alpha, double _beta, double _L1, double _L2, int _dim);
        double predict(vector<int>& x);
        void update(vector<int>& x, double p, int y);
};

FTRL::FTRL(double _alpha, double _beta, double _L1, double _L2, int _dim) {
    alpha = _alpha;
    beta = _beta;
    L1 = _L1;
    L2 = _L2;
    dim = _dim;
    n = vector<double>(dim, 0);
    z = vector<double>(dim, 0);
    w = vector<double>(dim, 0);
}

double FTRL::predict(vector<int>& x) {
    double wTx = 0;
    for (int i: x) {
        wTx += w[i];
    }
    // sigmoid function
    return sigmoid(wTx);
}

// update model parameters using one instance
void FTRL::update(vector<int>& x, double p, int y) {
    double g = p - y;  //x equals 1
    // update z and n
    for (int i: x) {
        double sigma = (sqrt(n[i] + g*g) - sqrt(n[i])) / alpha;
        z[i] += g - sigma*w[i];
        n[i] += g*g;
        // update weight
        int sign = sgn(z[i]); 
        if (sign*z[i] <= L1) {
            w[i] = 0;
        } else {
            w[i] = (sign*L1 - z[i]) / ((beta + sqrt(n[i]))/alpha + L2);
        }
    }
}

#endif  // FTRL_H_
