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
    void get_vystup();
    void get_aktiv();

    

    double o;
    double a;
    double delta;
    double der_akt_fun(double aktiv);
    std::vector<double> vstupy;
    std::vector<double> vahy;
    std::vector<double> vystupy_historie;
    std::vector<double> aktiv_historie;
    bool bias;
    enum {sigmoid} aktfunkce;
};

#endif // NEURON_H

