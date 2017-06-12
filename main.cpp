//
//  main.cpp
//  MultithreadLR
//
//  Created by 王资 on 17/6/10.
//  Copyright © 2017年 王资's. All rights reserved.
//

#include <iostream>
#include "/usr/local/include/armadillo"
#include <string>
#include <thread>
#include <time.h>
#include <stdio.h>
#include <vector>

using namespace std;
using arma::mat;

mat sigmoid(mat z) {
    return 1 / (1 + arma::exp(-z));
}


void normalize(mat& x) {
    double epthelon = 1e-20;
    for (int i = 0; i < x.n_cols; i++) {
        double std = arma::stddev(x.col(i));
        double mean = arma::mean(x.col(i));
        x.col(i) = (x.col(i) - mean) / (std + epthelon);
    }
}

double cost(mat& x, mat& y, mat& theta, double lambda) {
    using arma::dot;
    using arma::log;
    using arma::sum;
    double loss = sum(dot(-y, log(sigmoid(x * theta))) - dot((1 - y), log(1 - sigmoid(x * theta)))) / x.n_rows;
    double punish = arma::sum(arma::dot(theta, theta)) * lambda / (2 * theta.n_rows);
    return loss + punish;
}

void getSum(mat x, mat y, mat& theta, double lambda, mat& sum) {
//    cout << "getSum\n";
    mat temp(theta);
    temp.at(0) = 0;
    mat x_t = x.t();
//    cout << x_t.n_rows << ' ' << x_t.n_cols << endl;
    sum = x_t * (sigmoid(x * theta) - y) + lambda * temp;
//    cout << "getSum\n";
}

mat gradient(arma::mat& x, arma::mat& y, arma::mat& theta, double lambda) {
//    cout << "gradient\n";
    const int max_thread = 8;
    thread lt[max_thread];
    mat sums[max_thread];
    int batch = x.n_rows / max_thread;
    for (int i = 0; i < max_thread; i++) {
        int lo = i * batch;
        int hi = (i + 1) * batch < x.n_rows ? (i + 1) * batch : x.n_rows;
        sums[i] = arma::zeros(theta.n_rows, 1);
        lt[i] = thread(getSum, x.rows(lo, hi - 1), y.rows(lo, hi - 1), ref(theta), lambda, ref(sums[i]));
    }
    for (int i = 0; i < max_thread; i++)
        lt[i].join();
    mat sum = arma::zeros(theta.n_rows, 1);
    for (int i = 0; i < max_thread; i++)
        sum = sum + sums[i];
    return sum / x.n_rows;
}

void fit(mat& x, mat& y, mat& theta) {
    double alpha = .001;  //  learning rate
    double lambda = 1;     //  regluarizing
    
//    // batch-gradient version
    int max_iter = int(5e2);
    for (int i = 0; i < max_iter; i++) {
        theta = theta - alpha * gradient(x, y, theta, lambda);
        // to check whether cost goes down ever iterate
//         cout << cost(x, y, theta, lambda) << endl;
        
    }
    
    // mini-batch-gradient version
//    int max_iter = 10;
//    int mini_batch = 10000;
//    for (int j = 0; j < max_iter; j++) {
//        x = arma::shuffle(x, 0);
//        for (int i = 0; i < x.n_rows; i += mini_batch) {
//            int lo = i;
//            int hi = i < x.n_rows ? i : (x.n_rows - 1);
//            mat batch_x = x.rows(lo, hi);
//            mat batch_y = y.rows(lo, hi);
//            theta = theta - alpha * gradient(batch_x, batch_y, theta, lambda);
//            // to check whether cost goes down ever iterate
////             cout << cost(x, y, theta, lambda) << endl;
//        }
//    }
    
}

int main(int argc, const char * argv[]) {
    clock_t s;
    s = clock();
    std::cout << "loading data...\n";
    
    mat data_x;
    data_x.load("/Users/wangzi/PycharmProjects/Multi-threadLR/train_x.txt");
    mat data_y;
    data_y.load("/Users/wangzi/PycharmProjects/Multi-threadLR/train_y.txt");
    cout << "data load finished in " << (clock() - s) * 1.0 / CLOCKS_PER_SEC << " s\n";
    s = clock();
    
    unsigned long long N = data_x.n_rows;
    unsigned long long M = data_x.n_cols;
    mat train_x = data_x.rows(0, int(.8 * N) - 1);
    mat train_y = data_y.rows(0, int(.8 * N) - 1);
    mat cv_x = data_x.rows(int(.8 * N), N - 1);
    mat cv_y = data_y.rows(int(.8 * N), N - 1);
//    cout << train_x.n_cols << ' ' << train_x.n_rows << endl;
//    cout << train_y.n_cols << ' ' << train_y.n_rows << endl;
    mat theta = arma::zeros(M, 1);
    cout << "normalizing...\n";
    normalize(train_x);
    cout << "training...\n";
    fit(train_x, train_y, theta);
    cout << "train finished in " << (clock() - s) * 1.0 / CLOCKS_PER_SEC << " s\n";
    cout << "train error: " << cost(train_x, train_y, theta, 1) << endl;
    s = clock();
    
    cout << "cv testing...\n";
    normalize(cv_x);
    cout << "cv test error: " << cost(cv_x, cv_y, theta, 1) << endl;
    cout << "cv test finished in " << (clock() - s) * 1.0 / CLOCKS_PER_SEC << " s\n";
    return 0;
}
