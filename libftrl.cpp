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
#include <ctime>
#include <sys/timeb.h>
#include <algorithm>

void ftrl_learn(FTRL* ftrl, string trainfile) {
    struct timeb start, end;
    ftime(&start);
    ifstream fin(trainfile, ios::in);
    string line;
    int local_cnt = 0;
    while (getline(fin, line)) {
        vector<Entry> feas;
        int label = 0;
        if (!parseFeatureLine(line, label, feas)) return;
        local_cnt++;
        if (local_cnt % 1000000 == 0) {
            cout << "train " << local_cnt << "..." << flush;
        }
        // step1: predict
        double p = ftrl->predict(feas);
        // step2: update
        ftrl->update(feas, p, label);
    }
    fin.close();
    ftime(&end);
    int cost_sec = end.time - start.time;
    cout << "Cost time : " << cost_sec/60 << " m " << cost_sec%60 <<" s"<< endl;
}
// validation
void ftrl_test(FTRL* ftrl, string testfile) {
    struct timeb start, end;
    ftime(&start);
    ifstream fin(testfile, ios::in);
    string line;
    int local_cnt = 0;
    double loss = 0;
    while (getline(fin, line)) {
        vector<Entry> feas;
        int label = 0;
        if (!parseFeatureLine(line, label, feas)) return;
        local_cnt++;
        if (local_cnt % 1000000 == 0) {
            cout << "test " << local_cnt << "..." << flush;
        }
        // step1: predict
        double p = ftrl->predict(feas);
        loss += logloss(p, label);
    }
    cout << "test logloss: " << loss/local_cnt << endl;
    fin.close();
    ftime(&end);
}
// make predict using test file, and write prediction to result file
void ftrl_test(FTRL* ftrl, string testfile, string resfile) {
    struct timeb start, end;
    ftime(&start);
    ifstream fin(testfile, ios::in);
    ofstream fout(resfile, ios::out);
    string line;
    int local_cnt = 0;
    while (getline(fin, line)) {
        vector<Entry> feas;
        int label = 0;
        if (!parseFeatureLine(line, label, feas)) return;
        local_cnt++;
        if (local_cnt % 1000000 == 0) {
            cout << "test " << local_cnt << "..." << flush;
        }
        // step1: predict
        double p = ftrl->predict(feas);
        fout << p << endl;
    }
    fin.close();
    fout.close();
    ftime(&end);
}

void info() {
    cout << "./libftrl -trainfile trainfile -testfile testfile -output output -model modelfile -2d 1" << endl;
}

int main(int argc, char **argv) {
    if (argc == 1) {
        info();
        return 0;
    }
    double dim = pow(10, 7);
    bool interaction = false;
    double l1 = 0.1, l2 = 1.0;
    double alpha = 0.1, beta = 1.0;
    int epochs = 1;
    string trainfile, testfile, validfile, outputfile, modelfile;
    int i;
    if ((i = ArgPos((char *)"-train", argc, argv)) > 0) trainfile = string(argv[i + 1]);
    if ((i = ArgPos((char *)"-test", argc, argv)) > 0) testfile = string(argv[i + 1]);
    if ((i = ArgPos((char *)"-valid", argc, argv)) > 0) validfile = string(argv[i + 1]);
    if ((i = ArgPos((char *)"-output", argc, argv)) > 0) outputfile = string(argv[i + 1]);
    if ((i = ArgPos((char *)"-model", argc, argv)) > 0) modelfile = string(argv[i + 1]);
    if ((i = ArgPos((char *)"-2d", argc, argv)) > 0) interaction = stoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-l1", argc, argv)) > 0) l1 = stof(argv[i + 1]);
    if ((i = ArgPos((char *)"-l2", argc, argv)) > 0) l2 = stof(argv[i + 1]);
    if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = stof(argv[i + 1]);
    if ((i = ArgPos((char *)"-beta", argc, argv)) > 0) beta = stof(argv[i + 1]);
    if ((i = ArgPos((char *)"-epochs", argc, argv)) > 0) epochs = stoi(argv[i + 1]);

    FTRL* ftrl = new FTRL(alpha, beta, l1, l2, dim, interaction);
    ftrl->printModelParams();
    for (int i = 0; i < epochs; i++) {
        cout << "Epoch " << i << " begin!" << endl;
        ftrl_learn(ftrl, trainfile);
        if (!validfile.empty()) {
            ftrl_test(ftrl, validfile);
        }
        cout << "Epoch " << i << " End!" << endl;
    }
    ftrl_test(ftrl, testfile, outputfile);
    ftrl->saveModel(modelfile);
    return 0;
}

