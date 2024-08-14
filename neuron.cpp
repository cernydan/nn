#include "neuron.h"
#include <random>
#include <iostream>
#include <omp.h>

using namespace std;

Neuron::Neuron()    //konstruktor
{
    a = 0.0;
    o = 0.0;
    bias = true;
    aktfunkce = sigmoid;
    vstupy.clear();
    vahy.clear();
    vystupy_historie.clear();
}

Neuron::~Neuron()   // Destruktor
{
    a = 0.0;
    o = 0.0;
    bias = false;
    aktfunkce = sigmoid;
    vstupy.clear();
    vahy.clear();
    vystupy_historie.clear();
}

Neuron::Neuron(const Neuron& other) :   // Kopírovací konstruktor
    o(other.o),
    a(other.a),
    vstupy(other.vstupy),
    vahy(other.vahy),
    vystupy_historie(other.vystupy_historie),
    aktiv_historie(other.aktiv_historie),
    bias(other.bias),
    aktfunkce(other.aktfunkce) {}

Neuron::Neuron(Neuron&& other) noexcept :   // Move konstruktor
    o(std::move(other.o)),
    a(std::move(other.a)),
    vstupy(std::move(other.vstupy)),
    vahy(std::move(other.vahy)),
    vystupy_historie(std::move(other.vystupy_historie)),
    aktiv_historie(std::move(other.aktiv_historie)),
    bias(std::move(other.bias)),
    aktfunkce(std::move(other.aktfunkce)) {
    
    // Reset moved-from object
    other.o = 0.0;
    other.a = 0.0;
    other.bias = false;
    other.aktfunkce = sigmoid;
    other.vstupy.clear();
    other.vahy.clear();
    other.vystupy_historie.clear();
    other.aktiv_historie.clear();

}

Neuron& Neuron::operator=(const Neuron& other) {       // Kopírovací přiřazovací operátor
    if (this != &other) {  // Kontrola na přiřazení sebe sama
        o = other.o;
        a = other.a;
        vstupy = other.vstupy;
        vahy = other.vahy;
        vystupy_historie = other.vystupy_historie;
        aktiv_historie = other.aktiv_historie;
        bias = other.bias;
        aktfunkce = other.aktfunkce;
    }
    return *this;
}

Neuron& Neuron::operator=(Neuron&& other) noexcept {    // Move přiřazovací operátor
    if (this != &other) {  // Kontrola na přiřazení sebe sama
        o = std::move(other.o);
        a = std::move(other.a);
        vstupy = std::move(other.vstupy);
        vahy = std::move(other.vahy);
        vystupy_historie = std::move(other.vystupy_historie);
        aktiv_historie = std::move(other.aktiv_historie);
        bias = std::move(other.bias);
        aktfunkce = std::move(other.aktfunkce);

        // Reset moved-from object
        other.o = 0.0;
        other.a = 0.0;
        other.bias = false;
        other.aktfunkce = sigmoid;
        other.vstupy.clear();
        other.vahy.clear();
        other.vystupy_historie.clear();
        other.aktiv_historie.clear();

    }
    return *this;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Neuron::set_vstupy(const std::vector<double>& inputs) {
    vstupy = inputs;
    if(bias == true){
        vstupy.push_back(1.0);
    }
}

void Neuron::set_randomvahy(){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-0.3, 0.3);

    vahy.clear();
    for (int i = 0; i < vstupy.size(); ++i) {
        vahy.push_back(dis(gen));
    }
}

void Neuron::set_rucovahy(const std::vector<double>& weights){
    vahy.clear();
    vahy = weights;
}

void Neuron::vypocet() {
    a = 0.0;
    o = 0.0;
    double skalsoucprv = 0.0;

    for (int i = 0; i<vstupy.size();i++) {
        skalsoucprv = vstupy[i]*vahy[i];
        a = a + skalsoucprv;
    }
    
    switch (aktfunkce) {
        case sigmoid:
        o = 1/(1+exp(-a));
        break;
    }
    get_vystup();
    get_aktiv();
}

void Neuron::print_neuron() {
    std::cout <<"Vahy: ";
    for (double jednavaha : vahy) {
        std::cout<< jednavaha << " ";
    }
    std::cout <<"\n"<< "Aktivace: " <<a<< "\n";
    std::cout <<"Vystup: "<< o;

}

void Neuron::get_vystup() {
    vystupy_historie.push_back(o);
    if (vystupy_historie.size() > 5) {
        vystupy_historie.erase(vystupy_historie.begin());
    }
}

void Neuron::get_aktiv() {
    aktiv_historie.push_back(a);
    if (aktiv_historie.size() > 5) {
        aktiv_historie.erase(aktiv_historie.begin());
    }
}

double Neuron::der_akt_fun(double aktiv){
    switch (aktfunkce) {
        case sigmoid:
            return ((1/(1+exp(-aktiv)))*(1-(1/(1+exp(-aktiv)))));
        default:
            return aktiv;

    }
}
