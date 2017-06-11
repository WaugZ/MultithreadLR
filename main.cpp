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

mat getSum(mat& x, mat& y, mat& theta, double lambda) {
//    cout << "getSum\n";
    mat temp(theta);
    temp.at(0) = 0;
    mat x_t = x.t();
    mat sum = x_t * (sigmoid(x * theta) - y) + lambda * temp;
//    cout << "getSum\n";
    return sum;
}

mat gradient(arma::mat& x, arma::mat& y, arma::mat& theta, double lambda) {
    mat sum = getSum(x, y, theta, lambda);
    return sum / x.n_rows;
}

void fit(mat& x, mat& y, mat& theta) {
    double alpha = .001;  //  learning rate
    double lambda = 1;     //  regluarizing
    
//    // batch-gradient version
//    int max_iter = int(2e3);
//    for (int i = 0; i < max_iter; i++) {
//        theta = theta - alpha * gradient(x, y, theta, lambda);
//        // to check whether cost goes down ever iterate
//        // cout << cost(x, y, theta, lambda) << endl;
//        
//    }
    
    // mini-batch-gradient version
    int max_iter = 5;
    int mini_batch = 100;
    for (int j = 0; j < max_iter; j++) {
        for (int i = 0; i < x.n_rows; i += mini_batch) {
            int lo = i;
            int hi = i < x.n_rows ? i : (x.n_rows - 1);
            mat batch_x = x.rows(lo, hi);
            mat batch_y = y.rows(lo, hi);
            theta = theta - alpha * gradient(batch_x, batch_y, theta, lambda);
            // to check whether cost goes down ever iterate
             cout << cost(x, y, theta, lambda) << endl;
        }
    }
    
}

int main(int argc, const char * argv[]) {
    std::cout << "loading data...\n";
    
    mat data_x;
    data_x.load("/Users/wangzi/PycharmProjects/Multi-threadLR/train_x.txt");
    mat data_y;
    data_y.load("/Users/wangzi/PycharmProjects/Multi-threadLR/train_y.txt");
    
    unsigned long long N = data_x.n_rows;
    unsigned long long M = data_x.n_cols;
    mat train_x = data_x.rows(0, int(.8 * N) - 1);
    mat train_y = data_y.rows(0, int(.8 * N) - 1);
    mat cv_x = data_x.rows(int(.8 * N), N - 1);
    mat cv_y = data_y.rows(int(.8 * N), N - 1);
//    std::cout << train_x.n_cols << ' ' << train_x.n_rows << std::endl;
//    std::cout << train_y.n_cols << ' ' << train_y.n_rows << std::endl;
    mat theta = arma::zeros(M, 1);
    cout << "normalizing...\n";
    normalize(train_x);
    cout << "training...\n";
    fit(train_x, train_y, theta);
    
    return 0;
}
