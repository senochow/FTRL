/* ############################################################################
* 
* Copyright (c) 2016 ICT MCG Group, Inc. All Rights Reserved
* 
* ###########################################################################
* Brief: 
 
* Authors: zhouxing(@ict.ac.cn)
* Date:    2016/11/23 11:28:20
* File:    util.h
*/
#ifndef UTIL_H_
#define UTIL_H_
#include <iostream>
#include <cmath>
#include <string>
#include <cstring>
#include <sstream>
using namespace std;


inline double logloss(double p, int y) {
    double loss = (y == 1) ? -log(p) : -log(1-p);
    return loss;
}

inline double sigmoid(double x) {
    return 1.0 / (1 + exp(-x));
}

inline int sgn(double x) {
    return (x < 0) ? -1: 1;
}

inline void splitString(const string& s, char c, vector<string>& vec) {
    size_t i = 0;
    size_t j = s.find(c);
    while (j != string::npos) {
        vec.push_back(s.substr(i, j-i));
        i = ++j;
        j = s.find(c, j);
        if (j == string::npos) {
            vec.push_back(s.substr(i));
        }
    }
}

int getHash(string& name, int dim) {
    unsigned long long hash = 0;
    for (size_t i = 0; i < name.size(); i++) {
        hash = hash*257 + name[i];
    }
    hash = hash%dim;
    return hash;
}
int ArgPos(char *str, int argc, char **argv) {
    int a;
    for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
        if (a == argc - 1) {
            printf("Argument missing for %s\n", str);
            exit(1);
        }
        return a;
    }
    return -1;
}
#endif
