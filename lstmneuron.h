#pragma once
#ifndef LSTMNEURON_H
#define LSTMNEURON_H

#include "neuron.h"

class LSTMNeuron {
public:
    LSTMNeuron();    // Konstruktor
    ~LSTMNeuron();    // Destruktor
    LSTMNeuron(const LSTMNeuron& other);  // Kopírovací konstruktor
    LSTMNeuron(LSTMNeuron&& other) noexcept;  // Move konstruktor
    LSTMNeuron& operator=(const LSTMNeuron& other);  // Kopírovací přiřazovací operátor
    LSTMNeuron& operator=(LSTMNeuron&& other) noexcept;  // Move přiřazovací operátor

    Neuron forget;
    Neuron update;
    Neuron candidate;
    Neuron output;
    double shortterm,longterm,Wy,by,vystup;
    std::vector<double> forget_hist,update_hist,candidate_hist,output_hist,shortterm_hist,longterm_hist,vystup_hist;
    double dc = 0;
    double da = 0;
    double dLdz,dLda,dLdc,dLdcan,dLdu,dLdf,dLdo;
    std::vector<double> dLdx;


    void set_vstupy(const std::vector<double>& inputs);
    void set_randomvahy();
    void vypocet();
    
    double get_vystup();

};

#endif // LSTMNEURON_H