#include "lstmneuron.h"
#include <random>
#include <iostream>

using namespace std;

LSTMNeuron::LSTMNeuron()    //konstruktor
{
    forget.aktfunkce = Neuron::sigmoid;
    input_s.aktfunkce = Neuron::sigmoid;
    input_t.aktfunkce = Neuron::tanh;
    output.aktfunkce = Neuron::sigmoid;
    shortterm = 1.0;
    longterm= 1.0;
}

LSTMNeuron::~LSTMNeuron()   // Destruktor
{

}

LSTMNeuron::LSTMNeuron(const LSTMNeuron& other) :   // Kopírovací konstruktor
    forget(other.forget),
    input_s(other.input_s),
    input_t(other.input_t),
    output(other.output),
    shortterm(other.shortterm),
    longterm(other.longterm){}


LSTMNeuron::LSTMNeuron(LSTMNeuron&& other) noexcept :  // move konstruktor
    forget(std::move(other.forget)),
    input_s(std::move(other.input_s)),
    input_t(std::move(other.input_t)),
    output(std::move(other.output)),
    shortterm(std::move(other.shortterm)),
    longterm(std::move(other.longterm)) {

    // Reset moved-from object
    other.shortterm = 0.0;
    other.longterm = 0.0;
}


LSTMNeuron& LSTMNeuron::operator=(const LSTMNeuron& other) {        // Kopírovací přiřazovací operátor
    if (this != &other) {  // Kontrola na přiřazení sebe sama
        forget = other.forget;
        input_s = other.input_s;
        input_t = other.input_t;
        output = other.output;
        shortterm = other.shortterm;
        longterm = other.longterm;
    }
    return *this;
}


LSTMNeuron& LSTMNeuron::operator=(LSTMNeuron&& other) noexcept {       // Move přiřazovací operátor
    if (this != &other) {  // Kontrola na přiřazení sebe sama
        forget = std::move(other.forget);
        input_s = std::move(other.input_s);
        input_t = std::move(other.input_t);
        output = std::move(other.output);
        shortterm = std::move(other.shortterm);
        longterm = std::move(other.longterm);

        // Reset moved-from object
        other.shortterm = 0.0;
        other.longterm = 0.0;
    }
    return *this;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void LSTMNeuron::set_vstupy(const std::vector<double>& inputs){
    forget.bias = true;
    input_s.bias = true;
    input_t.bias = true;
    output.bias = true;
    forget.set_vstupy(inputs);
    input_s.set_vstupy(inputs);
    input_t.set_vstupy(inputs);
    output.set_vstupy(inputs);
    forget.vstupy.push_back(shortterm);
    input_s.vstupy.push_back(shortterm);
    input_t.vstupy.push_back(shortterm);
    output.vstupy.push_back(shortterm);          // vstupy,bias,shortterm
}

void LSTMNeuron::set_randomvahy(){
    forget.set_randomvahy();
    input_s.set_randomvahy();
    input_t.set_randomvahy();
    output.set_randomvahy();
}

void LSTMNeuron::vypocet(){
    forget.vypocet();
    input_s.vypocet();
    input_t.vypocet();
    output.vypocet();

    longterm = longterm * forget.get_vystup() + input_s.get_vystup() * input_t.get_vystup();
    shortterm = output.get_vystup() * (exp(longterm)-exp(-longterm))/(exp(longterm)+exp(-longterm));
}

double LSTMNeuron::get_vystup(){
    return shortterm;
}