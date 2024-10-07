#pragma once
#ifndef NEURON_H
#define NEURON_H

#include <vector>

class Neuron {
public:
    Neuron();    // Konstruktor
    ~Neuron();    // Destruktor
    Neuron(const Neuron& other);  // Kopírovací konstruktor
    Neuron(Neuron&& other) noexcept;  // Move konstruktor
    Neuron& operator=(const Neuron& other);  // Kopírovací přiřazovací operátor
    Neuron& operator=(Neuron&& other) noexcept;  // Move přiřazovací operátor

    void set_vstupy(const std::vector<double>& inputs);
    void vypocet();
    void set_randomvahy();
    void set_rucovahy(const std::vector<double>& weights);
    void print_neuron();
    
    double get_vystup();
    double get_aktiv();
    double der_akt_fun(double aktiv);

    double o;
    double a;
    double delta;
    bool bias;
    enum {sigmoid,linear,leakyrelu,tanh} aktfunkce;
    std::vector<double> Mt;
    std::vector<double> Mt_s;
    std::vector<double> Vt;
    std::vector<double> Vt_s;
    std::vector<double> vstupy;
    std::vector<double> vahy;

};

#endif // NEURON_H

