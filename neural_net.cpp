#include "neural_net.h"
#include "neuron.h"
#include <random>
#include <iostream>

using namespace std;

NN::NN()    //konstruktor
{
    pocet_vrstev = 0;
    rozmery.clear();
    sit.clear();
}

NN::~NN()   // Destruktor
{
    pocet_vrstev = 0;
    rozmery.clear();
    sit.clear();
}

NN::NN(const NN& other) :   // Kopírovací konstruktor
    
    pocet_vrstev(other.pocet_vrstev),
    rozmery(other.rozmery),
    sit(other.sit) {}

NN::NN(NN&& other) noexcept :   // Move konstruktor
    pocet_vrstev(std::move(other.pocet_vrstev)),
    rozmery(std::move(other.rozmery)),
    sit(std::move(other.sit))  {
    // Reset moved-from object
    other.pocet_vrstev = 0;
    other.rozmery.clear();
    other.sit.clear();
}

NN& NN::operator=(const NN& other) {    // Kopírovací přiřazovací operátor
    if (this != &other) {  // Kontrola na přiřazení sebe sama
        pocet_vrstev = other.pocet_vrstev;
        rozmery = other.rozmery;
        sit = other.sit;
    }
    return *this;
}

NN& NN::operator=(NN&& other) noexcept {   // Move přiřazovací operátor
    if (this != &other) {  // Kontrola na přiřazení sebe sama
        pocet_vrstev = std::move(other.pocet_vrstev);
        rozmery = std::move(other.rozmery);
        sit = std::move(other.sit);

        // Reset moved-from object
        other.pocet_vrstev = 0;
        other.rozmery.clear();
        other.sit.clear();
    }
    return *this;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////ú

void NN::set_rozmery(const std::vector<int>& rozmers) {
    rozmery = rozmers;
    pocet_vrstev = rozmery.size();
}

void NN::set_chtenejout(const std::vector<double>& obsout) {
    chtenejout = obsout;
}

void NN::print_nn(){
    std::cout <<"Pocet vrstev: "<< pocet_vrstev<<"\n";
    
    std::cout <<"Pocty neuronu ve vrstvach: ";
    for (int jednavrstva : rozmery) {
        std::cout<< jednavrstva << " ";
    }
}

void NN::set_train_data(const std::vector<std::vector<double>>& datas){
    train_data = datas;
}

void NN::set_val_data(const std::vector<std::vector<double>>& datas){
    val_data = datas;
}

//!/!/!/!/!/!/!/!/!/! UDELAT set a print i pro test a val data, nebo nejak vyber v 1 funkci???

void NN::print_data(){
    for (int i = 0; i<train_data.size();++i){
        for (int j = 0; j<train_data[i].size();j++){
            std::cout<<train_data[i][j]<<" ";
            if(j == (train_data[i].size()-1)){
                std::cout<<"\n";
            }
        }
    }
}

void NN::init_sit() {
    sit.clear();
    sit.resize(pocet_vrstev);
    for (int i = 0; i < pocet_vrstev; ++i) {
        sit[i].resize(rozmery[i]);
        for (int j = 0; j < rozmery[i]; ++j) {
            sit[i][j] = Neuron();
        }
    }

    pom_vystup.clear();
    vystupy.clear();
    for (int i = 0; i < rozmery[0]; ++i) {
        sit[0][i].set_vstupy (train_data[0]);
        sit[0][i].set_randomvahy();
        sit[0][i].vypocet();
        pom_vystup.push_back(sit[0][i].o);
    }

    for (int i = 1; i < pocet_vrstev; ++i) {
        for (int j = 0; j < rozmery[i]; ++j) {
            sit[i][j].set_vstupy(pom_vystup);
            sit[i][j].set_randomvahy();
            sit[i][j].vypocet();
        }
        pom_vystup.clear();
        for (int j = 0; j < rozmery[i]; ++j) {
            pom_vystup.push_back( sit[i][j].o);
        }
    }
}

void NN::online_bp(int iter){
    for(int m = 0;m<iter;++m){
        std::cout<<m<<"\n";
    vystupy.clear();
    pom_vystup.clear();
    for (int l = 0;l<train_data.size();++l){
        for (int i = 0; i < rozmery[0]; ++i) {
            sit[0][i].set_vstupy (train_data[l]);
            sit[0][i].vypocet();
            pom_vystup.push_back(sit[0][i].o);
        }
        
        for (int i = 1; i < pocet_vrstev; ++i) {
            for (int j = 0; j < rozmery[i]; ++j) {
                sit[i][j].set_vstupy(pom_vystup);
                sit[i][j].vypocet();
            }
            pom_vystup.clear();
            for (int j = 0; j < rozmery[i]; ++j) {
                pom_vystup.push_back( sit[i][j].o);
            }
        }
        vystupy.push_back(pom_vystup[0]);
                            // dava smysl jen pro 1 vystup


        sit[pocet_vrstev-1][rozmery[pocet_vrstev-1]-1].delta = vystupy[l] - chtenejout[l];
        for (int i=0;i<rozmery[pocet_vrstev-2];++i){
            sit[pocet_vrstev-2][i].delta = sit[pocet_vrstev-2][i].der_akt_fun(sit[pocet_vrstev-2][i].a)*(sit[pocet_vrstev-1][rozmery[pocet_vrstev-1]-1].vahy[i] * sit[pocet_vrstev-1][rozmery[pocet_vrstev-1]-1].delta);
         }

        for(int j = (pocet_vrstev-3); j>=0;--j){
            for (int i=0;i<rozmery[j];++i){
                 double skalsoucprv = 0.0;
                for(int k = 0;k<rozmery[j+1];++k){
                    skalsoucprv += sit[j+1][k].vahy[i] * sit[j+1][k].delta;
                }
            
            sit[j][i].delta = sit[j][i].der_akt_fun(sit[j][i].a)*skalsoucprv;
        }
    }

    for(int i = 0;i<pocet_vrstev;++i){
        for(int j = 0;j<rozmery[i];++j){
            for(int k = 0; k < sit[i][j].vahy.size();++k)
                sit[i][j].vahy[k] = sit[i][j].vahy[k] - alfa * sit[i][j].delta * sit[i][j].vstupy[k];
        }
    }

                            
    }
}}

void NN::valid(){
    vystupy.clear();
    pom_vystup.clear();
    for (int l = 0;l<val_data.size();++l){
        for (int i = 0; i < rozmery[0]; ++i) {
            sit[0][i].set_vstupy (val_data[l]);
            sit[0][i].vypocet();
            pom_vystup.push_back(sit[0][i].o);
        }
        
        for (int i = 1; i < pocet_vrstev; ++i) {
            for (int j = 0; j < rozmery[i]; ++j) {
                sit[i][j].set_vstupy(pom_vystup);
                sit[i][j].vypocet();
            }
            pom_vystup.clear();
            for (int j = 0; j < rozmery[i]; ++j) {
                pom_vystup.push_back( sit[i][j].o);
            }
        }
        vystupy.push_back(pom_vystup[0]);
                            // dava smysl jen pro 1 vystup
    }
}

void NN::count_cost(){
    cost = 0.0;
    for(int i=0;i<vystupy.size();++i){
        cost += (vystupy[i]-chtenejout[i])*(vystupy[i]-chtenejout[i]);   
    }
    cost = cost/vystupy.size();
}
