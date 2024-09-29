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
    Neuron input_s;
    Neuron input_t;
    Neuron output;
    double shortterm;
    double longterm;

    void set_vstupy(const std::vector<double>& inputs);
    void set_randomvahy();
    void vypocet();
    
    double get_vystup();

};

#endif // LSTMNEURON_H