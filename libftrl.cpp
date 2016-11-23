/* ############################################################################
* 
* Copyright (c) 2016 ICT MCG Group, Inc. All Rights Reserved
* 
* ###########################################################################
* Brief: 
 
* Authors: zhouxing(@ict.ac.cn)
* Date:    2016/11/23 10:23:06
* File:    libftrl.cpp
*/
#include "ftrl_v1.h"
#include <fstream>
#include <ctime>
#include <sys/timeb.h>

struct Instance {
    vector<int> x;
    int y;
    string id;
    int date;
    Instance(vector<string>& feas, vector<string>& keyname, int dim) {
        id = feas[0];
        y = stoi(feas[1]);
        date = stoi(feas[2].substr(4, 2));
        feas[2] = feas[2].substr(6);
        for (size_t i = 2; i < feas.size(); i++) {
            string key = keyname[i] + "_" + feas[i];
            int value = getHash(key, dim);
            x.push_back(value);
        }
    }

};


void ftrl_learn(FTRL* ftrl, string trainfile) {
    struct timeb start, end;
    ftime(&start);
    ifstream fin(trainfile, ios::in);
    string line;
    // pass first line
    getline(fin, line);
    vector<string> keyname;
    splitString(line, ',', keyname);
    double loss = 0;
    int count = 0;
    int local_cnt = 0;
    while (getline(fin, line)) {
        vector<string> feas;
        splitString(line, ',', feas);
        Instance inst(feas, keyname, ftrl->dim);
        local_cnt++;
        if (local_cnt % 1000000 == 0) {
            cout << "train " << local_cnt << endl;
        }

        // step1: predict
        double p = ftrl->predict(inst.x);
        if (inst.date > 29) {
            loss += logloss(p, inst.y);
            count++;
        } else {
            // step2: update
            ftrl->update(inst.x, p, inst.y);
        }
    }
    fin.close();
    cout << "validation logloss: " << loss/count << endl;
    ftime(&end);
    int cost_sec = end.time - start.time;
    cout << "Cost time : " << cost_sec/60 << " m " << cost_sec%60 <<" s"<< endl;
}
void ftrl_prediction(FTRL* ftrl, string testfile, string resfile) {
    struct timeb pstart, pend;
    ftime(&pstart);
    ifstream fin(testfile, ios::in); 
    ofstream fout(resfile, ios::out);
    fout << "id,click" << endl;
    string line;
    // pass first line
    getline(fin, line);
    vector<string> keyname;
    splitString(line, ',', keyname);
    while (getline(fin, line)) {
        vector<string> feas;
        splitString(line, ',', feas);
        feas[1] = feas[1].substr(6);
        vector<int> x;
        // test feature start from index 1
        for (size_t i = 1; i < feas.size(); i++) {
            string key = keyname[i] + "_" + feas[i];
            int value = getHash(key, ftrl->dim);
            x.push_back(value);
        }
        double p = ftrl->predict(x);
        fout << feas[0] << "," << p << endl;
    }
    fout.close();
    fin.close();
    ftime(&pend);
    int cost_sec = pend.time - pstart.time;
    cout << "Cost time : " << cost_sec/60 << " m " << cost_sec%60 <<" s"<< endl;
}
void info() {
    cout << "./libftrl -trainfile trainfile -testfile testfile -output output" << endl;
}

int main(int argc, char **argv) {
    if (argc == 1) {
        info();
        return 0;
    }
    double dim = pow(2, 26);
    string trainfile, testfile, outputfile;
    int i;
    if ((i = ArgPos((char *)"-train", argc, argv)) > 0) trainfile = string(argv[i + 1]);
    if ((i = ArgPos((char *)"-test", argc, argv)) > 0) testfile = string(argv[i + 1]);
    if ((i = ArgPos((char *)"-output", argc, argv)) > 0) outputfile = string(argv[i + 1]);
    FTRL* ftrl = new FTRL(0.05, 1.0, 1, 1.0, dim);
    int epochs = 3;
    for (int i = 0; i < epochs; i++) {
        ftrl_learn(ftrl, trainfile);
    }
    ftrl_prediction(ftrl, testfile, outputfile);
    return 0;
}

