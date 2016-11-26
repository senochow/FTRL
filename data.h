/* ############################################################################
* 
* Copyright (c) 2016 ICT MCG Group, Inc. All Rights Reserved
* 
* ###########################################################################
* Brief: 
 
* Authors: zhouxing(@ict.ac.cn)
* Date:    2016/11/26 14:36:13
* File:    data.h
*/
#ifndef DATA_H_
#define DATA_H_

#include <iostream>
#include <string>
#include <cstring>
using namespace std;


struct Entry {
    int id;
    int value;
    Entry (string s, char c) {
        size_t i = 0;
        size_t j = s.find(c);
        id = stoi(s.substr(i, j-i));
        value = stoi(s.substr(j+1));
    }
};

#endif
