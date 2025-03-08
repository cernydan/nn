#include "neuron.h"
#include <random>
#include <iostream>
#include <omp.h>

using namespace std;

Neuron::Neuron()    //konstruktor
{
    a = 0.0;
    o = 0.0;
    delta = 0.0;
    bias = true;
    aktfunkce = sigmoid;
    Mt.clear();
    Mt_s.clear();
    Vt.clear();
    Vt_s.clear();
    vstupy.clear();
    vahy.clear();
}

Neuron::~Neuron()   // Destruktor
{
    Mt.clear();
    Mt_s.clear();
    Vt.clear();
    Vt_s.clear();
    vstupy.clear();
    vahy.clear();
}

Neuron::Neuron(const Neuron& other) :   // Kopírovací konstruktor
    o(other.o),
    a(other.a),
    delta(other.delta),
    bias(other.bias),
    aktfunkce(other.aktfunkce),
    Mt(other.Mt),
    Mt_s(other.Mt_s),
    Vt(other.Vt),
    Vt_s(other.Vt_s),
    vstupy(other.vstupy),
    vahy(other.vahy){}


Neuron::Neuron(Neuron&& other) noexcept :       //Move konstruktor
    o(std::move(other.o)),
    a(std::move(other.a)),
    delta(std::move(other.delta)),
    bias(std::move(other.bias)),
    vstupy(std::move(other.vstupy)),
    vahy(std::move(other.vahy)),
    aktfunkce(std::move(other.aktfunkce)),
    Mt(std::move(other.Mt)),
    Mt_s(std::move(other.Mt_s)),
    Vt(std::move(other.Vt)),
    Vt_s(std::move(other.Vt_s)) {

    // Reset moved-from object
    other.o = 0.0;
    other.a = 0.0;
    other.delta = 0.0;
    other.bias = false;
    other.aktfunkce = sigmoid;
    other.vstupy.clear();
    other.vahy.clear();
    other.Mt.clear();
    other.Mt_s.clear();
    other.Vt.clear();
    other.Vt_s.clear();
}


Neuron& Neuron::operator=(const Neuron& other) {        // Kopírovací přiřazovací operátor
    if (this != &other) {  // Kontrola na přiřazení sebe sama
        o = other.o;
        a = other.a;
        delta = other.delta;
        bias = other.bias;
        vstupy = other.vstupy;
        vahy = other.vahy;
        aktfunkce = other.aktfunkce;
        Mt = other.Mt;
        Mt_s = other.Mt_s;
        Vt = other.Vt;
        Vt_s = other.Vt_s;
    }
    return *this;
}


Neuron& Neuron::operator=(Neuron&& other) noexcept {        // Move přiřazovací operátor
    if (this != &other) {  // Kontrola na přiřazení sebe sama
        o = std::move(other.o);
        a = std::move(other.a);
        delta = std::move(other.delta);
        bias = std::move(other.bias);
        vstupy = std::move(other.vstupy);
        vahy = std::move(other.vahy);
        aktfunkce = std::move(other.aktfunkce);
        Mt = std::move(other.Mt);
        Mt_s = std::move(other.Mt_s);
        Vt = std::move(other.Vt);
        Vt_s = std::move(other.Vt_s);

        // Reset moved-from object
        other.o = 0.0;
        other.a = 0.0;
        other.delta = 0.0;
        other.bias = false;
        other.aktfunkce = sigmoid;
        other.vstupy.clear();
        other.vahy.clear();
        other.Mt.clear();
        other.Mt_s.clear();
        other.Vt.clear();
        other.Vt_s.clear();
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
    std::normal_distribution<> dis(0.0,0.1);

         vahy.clear();
        for (int i = 0; i < vstupy.size(); ++i) {
            vahy.push_back(dis(gen));
            Mt.push_back(0.0);
            Mt_s.push_back(0.0);
            Vt.push_back(0.0);
            Vt_s.push_back(0.0);
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

        case linear:
        o = a;
        break;

        case relu:

        if(0.0>a){
            o = 0.0;
        } else{
            o = a;
        }
        break;

        case leakyrelu:
        if((0.01*a)>a){
            o = 0.01*a;
        } else{
            o = a;
        }
        break;

        case tanh:
        o = (exp(a)-exp(-a))/(exp(a)+exp(-a));

        break;
    }
}

void Neuron::print_neuron() {
    std::cout <<"Vahy: ";
    for (double jednavaha : vahy) {
        std::cout<< jednavaha << " ";
    }
    std::cout <<"\n"<< "Aktivace: " <<a<< "\n";
    std::cout <<"Vystup: "<< o;

}

double Neuron::get_vystup() {
    return o;
}

double Neuron::get_aktiv() {
    return a;
}

double Neuron::der_akt_fun(double aktiv){
    switch (aktfunkce) {
        case sigmoid:
            return ((1/(1+exp(-aktiv)))*(1-(1/(1+exp(-aktiv)))));

        case linear:
            return (1.0);

        case leakyrelu:
            if(aktiv<0.0){ 
                return 0.01;
            } else {
                return 1.0;
            }
        
        case relu:
            if(aktiv<0.0){ 
                return 0.0;
            } else {
                return 1.0;
            }

        case tanh:
            return (1-pow(((exp(aktiv)-exp(-aktiv))/(exp(aktiv)+exp(-aktiv))),2));

        default:
            return aktiv;

    }
}
