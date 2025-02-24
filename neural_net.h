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

    void set_vstup_rady(const std::vector<double>& Qkal_in, const std::vector<double>& Qval_in,
                        const std::vector<double>& Rkal_in, const std::vector<double>& Rval_in,
                        const std::vector<double>& Tkal_in, const std::vector<double>& Tval_in);
    Matice<double> udelej_radky(size_t velrad, const std::vector<double>& cr);
    Matice<double> udelej_lag(size_t lag,const std::vector<double>& cr);
    Tenzor<double> max_pool_fullstep_3d(Tenzor<double> vstupnim, size_t oknorad, size_t oknosl);
    double tanh(double x);

    void cnn_full_cal(int iter);
    void cnn_full_val();
    void cnn1D_cal(size_t vel_ker, size_t poc_ker, int iter, int velic);
    void cnn1D_val(int velic);
    void cnnonfly_cal(size_t row_ker, size_t col_ker, size_t poc_ker, int iter);
    void cnnonfly_val();
    Tenzor<double> kernely_onfly;
    Tenzor<double> kernely_1D;
    Matice<double> biaskonv_1D;
    std::vector<double> biaskonv_onfly;

    Tenzor<double> kernely_full_1_Q;
    Tenzor<double> kernely_full_2_Q;
    std::vector<double> bias_full_k1_Q;
    std::vector<double> bias_full_k2_Q;
    Tenzor<double> kernely_full_1_R;
    Tenzor<double> kernely_full_2_R;
    std::vector<double> bias_full_k1_R;
    std::vector<double> bias_full_k2_R;
    Tenzor<double> kernely_full_1_T;
    Tenzor<double> kernely_full_2_T;
    std::vector<double> bias_full_k1_T;
    std::vector<double> bias_full_k2_T;



    std::vector<double> Q_kal_vstup;
    std::vector<double> R_kal_vstup;
    std::vector<double> T_kal_vstup;
    std::vector<double> Q_val_vstup;
    std::vector<double> R_val_vstup;
    std::vector<double> T_val_vstup;
};

#endif // NEURAL_NET_H