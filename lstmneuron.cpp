#include "lstmneuron.h"
#include <random>
#include <iostream>

using namespace std;

LSTMNeuron::LSTMNeuron()    //konstruktor
{
    forget.aktfunkce = Neuron::sigmoid;
    update.aktfunkce = Neuron::sigmoid;
    candidate.aktfunkce = Neuron::tanh;
    output.aktfunkce = Neuron::sigmoid;
    shortterm = 1.0;
    longterm= 1.0;
}

LSTMNeuron::~LSTMNeuron()   // Destruktor
{

}

LSTMNeuron::LSTMNeuron(const LSTMNeuron& other) :   // Kopírovací konstruktor
    forget(other.forget),
    update(other.update),
    candidate(other.candidate),
    output(other.output),
    shortterm(other.shortterm),
    longterm(other.longterm){}


LSTMNeuron::LSTMNeuron(LSTMNeuron&& other) noexcept :  // move konstruktor
    forget(std::move(other.forget)),
    update(std::move(other.update)),
    candidate(std::move(other.candidate)),
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
        update = other.update;
        candidate = other.candidate;
        output = other.output;
        shortterm = other.shortterm;
        longterm = other.longterm;
    }
    return *this;
}


LSTMNeuron& LSTMNeuron::operator=(LSTMNeuron&& other) noexcept {       // Move přiřazovací operátor
    if (this != &other) {  // Kontrola na přiřazení sebe sama
        forget = std::move(other.forget);
        update = std::move(other.update);
        candidate = std::move(other.candidate);
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
    update.bias = true;
    candidate.bias = true;
    output.bias = true;
    forget.set_vstupy(inputs);
    update.set_vstupy(inputs);
    candidate.set_vstupy(inputs);
    output.set_vstupy(inputs);
    forget.vstupy.push_back(shortterm);
    update.vstupy.push_back(shortterm);
    candidate.vstupy.push_back(shortterm);
    output.vstupy.push_back(shortterm);          // vstupy,bias,shortterm
}

void LSTMNeuron::set_randomvahy(){
    forget.set_randomvahy();
    update.set_randomvahy();
    candidate.set_randomvahy();
    output.set_randomvahy();
    Wy = 1;
    by = 0.3;

}

void LSTMNeuron::vypocet(){
    forget.vypocet();
    update.vypocet();
    candidate.vypocet();
    output.vypocet();

    longterm = longterm * forget.get_vystup() + update.get_vystup() * candidate.get_vystup();
    shortterm = output.get_vystup() * (exp(longterm)-exp(-longterm))/(exp(longterm)+exp(-longterm));
    vystup = shortterm * Wy + by;
    
    forget_hist.push_back(forget.o);
    update_hist.push_back(update.o);
    candidate_hist.push_back(candidate.o);
    output_hist.push_back(output.o);
    shortterm_hist.push_back(shortterm);
    longterm_hist.push_back(longterm);
    vystup_hist.push_back(vystup);
}



double LSTMNeuron::get_vystup(){
    return vystup;
}