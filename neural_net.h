#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include <vector>
#include <iostream>
#include <memory>
#include "neuron.h"

class NN{
public:
    NN();   //konstruktor
    ~NN();  // Destruktor
    NN(const NN& other);  // Kopírovací konstruktor
    NN(NN&& other) noexcept;  // Move konstruktor
    NN& operator=(const NN& other);  // Kopírovací přiřazovací operátor
    NN& operator=(NN&& other) noexcept;  // Move přiřazovací operátor

    void set_rozmery(const std::vector<int>& inputs);
    void print_nn();
    void init_sit();
    void set_train_data(const std::vector<std::vector<double>>& datas);
    void print_data();
    void online_bp(int iter);
    void set_chtenejout(const std::vector<double>& obsout);
    void set_val_data(const std::vector<std::vector<double>>& datas);
    void valid();
    void count_cost();

    int pocet_vrstev;
    std::vector<int> rozmery;
    std::vector<std::vector<Neuron>> sit;
    std::vector<std::vector<double>> train_data;
    std::vector<std::vector<double>> test_data;
    std::vector<std::vector<double>> val_data;
    std::vector<double> pom_vystup;
    std::vector<double> vystupy;
    std::vector<double> chtenejout;
    double cost;
    double alfa = 0.01;
};

#endif // NEURAL_NET_H