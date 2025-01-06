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
    std::vector<int> rozmery;
    std::vector<std::vector<Neuron>> sit;
    std::vector<double> pom_vystup;
    std::vector<double> vystupy;
    std::vector<double> chtenejout;

    std::vector<std::vector<double>> train_data;
    std::vector<std::vector<double>> val_data;
    
    Matice<double> konvo(Matice<double> vstupnim, Matice<double> vstupkernel);
    Tenzor<double> konvo_3d(Tenzor<double> vstupnim, Tenzor<double> vstupkernel);
    Matice<double> konvo_fullstep(Matice<double> vstupnim, Matice<double> vstupkernel);
    Tenzor<double> konvo_fullstep_3d(Tenzor<double> vstupnt, Tenzor<double> vstupker);
    Tenzor<double> konvo_fullstep_3d_1by1(Tenzor<double> vstupnt, Tenzor<double> vstupker);

    Matice<double> dataprocnn;

    void print_vystup();
    void print_nn();
    void init_sit(int poc_vstupu, const std::vector<int>& rozmers);

    void online_bp(int iter);
    void online_bp_adam(int iter);
    void set_chtenejout(const std::vector<double>& obsout);
    void valid();
    void count_cost();

    void set_train_data(const std::vector<std::vector<double>>& datas);
    void set_val_data(const std::vector<std::vector<double>>& datas);
    void print_data();
    void shuffle_train();

    void set_vstup_rada(const std::vector<double>& inputs);
    Matice<double> udelej_radky(size_t velrad, const std::vector<double>& cr);
    Matice<double> udelej_lag(size_t lag,const std::vector<double>& cr);
    double tanh(double x);

    void cnn_full(int iter);
    void cnn1D_cal(size_t vel_ker, size_t poc_ker, int iter);
    void cnnonfly_cal(size_t vel_ker, size_t poc_ker, int iter);
    void cnnonfly_val();
    Tenzor<double> kernely_onfly;
    Tenzor<double> kernely_1D;

    Tenzor<double> kernely_full_1;
    Tenzor<double> kernely_full_2s;
    Tenzor<double> kernely_full_3;
    Tenzor<double> kernely_full_4s;


    std::vector<double> Q_kal_vstup;
    std::vector<double> R_kal_vstup;
    std::vector<double> T_kal_vstup;
    std::vector<double> Q_val_vstup;
    std::vector<double> R_val_vstup;
    std::vector<double> T_val_vstup;
};

#endif // NEURAL_NET_H