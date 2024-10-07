#pragma once
#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include <vector>
#include <iostream>
#include <memory>
#include "neuron.h"
#include "matice.h"
#include "tenzor.h"
#include "lstmneuron.h"
#include "threadpool.h"

class NN{
public:
    NN();   //konstruktor
    ~NN();  // Destruktor
    NN(const NN& other);  // Kopírovací konstruktor
    NN(NN&& other) noexcept;  // Move konstruktor
    NN& operator=(const NN& other);  // Kopírovací přiřazovací operátor
    NN& operator=(NN&& other) noexcept;  // Move přiřazovací operátor

    int pocet_vrstev;
    double cost;
    double alfa;
    double beta;
    double beta2;
    double epsi;
    enum Co {radky,lag};
    std::vector<int> rozmery;
    std::vector<std::vector<Neuron>> sit;
    std::vector<std::vector<LSTMNeuron>> lstm_sit;
    std::vector<double> pom_vystup;
    std::vector<double> vystupy;
    std::vector<double> chtenejout;

    std::vector<std::vector<double>> train_data;
    std::vector<std::vector<double>> test_data;
    std::vector<std::vector<double>> val_data;
    
    Matice<double> set_kernel_rand(size_t radky, size_t sloupce);
    Matice<double> max_pool(Matice<double> vstupnim, size_t oknorad, size_t oknosl);
    Matice<double> avg_pool(Matice<double> vstupnim, size_t oknorad, size_t oknosl);
    Matice<double> konvo(Matice<double> vstupnim, Matice<double> vstupkernel);
    std::vector<double> vstupni_cr;
    std::vector<Matice<double>> kernely_v2d;
    std::vector<Matice<double>> dataprocnn_v2d;

    Tenzor<double> dataprocnn_t;
    Tenzor<double> kernely_t;

    void print_vystup();
    void print_nn();
    void init_sit(int poc_vstupu, const std::vector<int>& rozmers);
    void init_lstm(int poc_vstupu, const std::vector<int>& rozmers);
    void online_lstm(int iter);
    void lstm_1cell(int batch_size, int iter);
    void online_bp(int iter);
    void online_bp_adam(int iter);
    void online_bp_thread(int iter);
    void online_bp_th(int iter);
    void set_chtenejout(const std::vector<double>& obsout);
    void valid();
    void count_cost();
    void cnn_pokus(int iter);

    void set_train_data(const std::vector<std::vector<double>>& datas);
    void set_val_data(const std::vector<std::vector<double>>& datas);
    void print_data();
    void shuffle_train();

    void set_vstup_rada(const std::vector<double>& inputs);
    void udelej_radky(size_t velrad, bool tenzor);
    void udelej_lag(size_t lag, bool tenzor);
    void udelej_api(int n, double beta, Co coze, int kolik, bool tenzor);
    void udelej_prumery(int n, Co coze, int kolik, bool tenzor);
    double tanh(double x);

    

};

#endif // NEURAL_NET_H