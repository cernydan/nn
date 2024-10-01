#include "neural_net.h"
#include <random>
#include <iostream>
#include <cmath>
#include <algorithm>

using namespace std;

NN::NN()    //konstruktor
{
    pocet_vrstev = 0;
    cost = 0;
    alfa = 0.001;
    beta = 0.9;
    beta2 = 0.99;
    epsi = 0.00000001; 
    rozmery.clear();
    sit.clear();
    train_data.clear();
    test_data.clear();
    val_data.clear();
    pom_vystup.clear();
    vystupy.clear();
    chtenejout.clear();
    kernely_v2d.clear();
    vstupni_cr.clear();
    dataprocnn_v2d.clear();
}

NN::~NN()   // Destruktor
{
    rozmery.clear();
    sit.clear();
    train_data.clear();
    test_data.clear();
    val_data.clear();
    pom_vystup.clear();
    vystupy.clear();
    chtenejout.clear();
    kernely_v2d.clear();
    vstupni_cr.clear();
    dataprocnn_v2d.clear();
}

NN::NN(const NN& other) :   // Kopírovací konstruktor
    
    pocet_vrstev(other.pocet_vrstev),
    cost(other.cost),
    alfa(other.alfa),
    beta(other.beta),
    beta2(other.beta2),
    epsi(other.epsi),
    rozmery(other.rozmery),
    sit(other.sit),
    train_data(other.train_data),
    test_data(other.test_data),
    val_data(other.val_data),
    pom_vystup(other.pom_vystup),
    vystupy(other.vystupy),
    chtenejout(other.chtenejout),
    kernely_v2d(other.kernely_v2d),
    vstupni_cr(other.vstupni_cr),
    dataprocnn_v2d(other.dataprocnn_v2d) {}

NN::NN(NN&& other) noexcept :   // Move konstruktor
    pocet_vrstev(std::move(other.pocet_vrstev)),
    cost(std::move(other.cost)),
    alfa(std::move(other.alfa)),
    beta(std::move(other.beta)),
    beta2(std::move(other.beta2)),
    epsi(std::move(other.epsi)),
    rozmery(std::move(other.rozmery)),
    sit(std::move(other.sit)),
    train_data(std::move(other.train_data)),
    test_data(std::move(other.test_data)),
    val_data(std::move(other.val_data)),
    pom_vystup(std::move(other.pom_vystup)),
    vystupy(std::move(other.vystupy)),
    chtenejout(std::move(other.chtenejout)),
    kernely_v2d(std::move(other.kernely_v2d)),
    vstupni_cr(std::move(other.vstupni_cr)),
    dataprocnn_v2d(std::move(other.dataprocnn_v2d)) {

    // Reset moved-from object
    other.pocet_vrstev = 0;
    other.cost = 0;
    other.alfa = 0.001;
    other.beta = 0.9;
    other.beta2 = 0.99;
    other.epsi = 0.00000001;
    other.rozmery.clear();
    other.sit.clear();
    other.train_data.clear();
    other.test_data.clear();
    other.val_data.clear();
    other.pom_vystup.clear();
    other.vystupy.clear();
    other.chtenejout.clear();
    other.kernely_v2d.clear();
    other.vstupni_cr.clear();
    other.dataprocnn_v2d.clear();
}

NN& NN::operator=(const NN& other) {    // Kopírovací přiřazovací operátor
    if (this != &other) {  // Kontrola na přiřazení sebe sama
        pocet_vrstev = other.pocet_vrstev;
        cost = other.cost;
        alfa = other.alfa;
        beta = other.beta;
        beta2 = other.beta2;
        epsi = other.epsi;
        rozmery = other.rozmery;
        sit = other.sit;
        train_data = other.train_data;
        test_data = other.test_data;
        val_data = other.val_data;
        pom_vystup = other.pom_vystup;
        vystupy = other.vystupy;
        chtenejout = other.chtenejout;
        kernely_v2d = other.kernely_v2d;
        vstupni_cr = other.vstupni_cr;
        dataprocnn_v2d = other.dataprocnn_v2d;
    }
    return *this;
}

NN& NN::operator=(NN&& other) noexcept {   // Move přiřazovací operátor
    if (this != &other) {  // Kontrola na přiřazení sebe sama
        pocet_vrstev = std::move(other.pocet_vrstev);
        cost = std::move(other.cost);
        alfa = std::move(other.alfa);
        beta = std::move(other.beta);
        beta2 = std::move(other.beta2);
        epsi = std::move(other.epsi);
        rozmery = std::move(other.rozmery);
        sit = std::move(other.sit);
        train_data = std::move(other.train_data);
        test_data = std::move(other.test_data);
        val_data = std::move(other.val_data);
        pom_vystup = std::move(other.pom_vystup);
        vystupy = std::move(other.vystupy);
        chtenejout = std::move(other.chtenejout);
        kernely_v2d = std::move(other.kernely_v2d);
        vstupni_cr = std::move(other.vstupni_cr);
        dataprocnn_v2d = std::move(other.dataprocnn_v2d);

        // Reset moved-from object
        other.pocet_vrstev = 0;
        other.cost = 0;
        other.alfa = 0.001;
        other.beta = 0.9;
        other.beta2 = 0.99;
        other.epsi = 0.00000001;
        other.rozmery.clear();
        other.sit.clear();
        other.train_data.clear();
        other.test_data.clear();
        other.val_data.clear();
        other.pom_vystup.clear();
        other.vystupy.clear();
        other.chtenejout.clear();
        other.kernely_v2d.clear();
        other.vstupni_cr.clear();
        other.dataprocnn_v2d.clear();
    }
    return *this;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////ú

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

void NN::shuffle_train() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::vector<size_t> indexy(train_data.size());
    for (size_t i = 0; i < train_data.size(); ++i) {
        indexy[i] = i;
    }
    std::shuffle(indexy.begin(), indexy.end(), gen);
    std::vector<std::vector<double>> new_data(train_data.size());
    std::vector<double> new_vec(chtenejout.size());

    for (size_t i = 0; i < train_data.size(); ++i) {
        new_data[i] = train_data[indexy[i]];
        new_vec[i] = chtenejout[indexy[i]];
    }

    train_data = new_data;
    chtenejout = std::move(new_vec);

    new_data.clear();
    new_vec.clear();

}


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

void NN::init_sit(int poc_vstupu, const std::vector<int>& rozmers) {
    rozmery = rozmers;
    pocet_vrstev = rozmery.size();
    sit.clear();
    sit.resize(pocet_vrstev);
    for (int i = 0; i < pocet_vrstev; ++i) {
        sit[i].resize(rozmery[i]);
        for (int j = 0; j < rozmery[i]; ++j) {
            sit[i][j] = Neuron();
        }
    }
    for (int i = 0;i<sit[pocet_vrstev-1].size();++i){
        sit[pocet_vrstev-1][i].aktfunkce = Neuron::linear;
    }
    pom_vystup.clear();
    vystupy.clear();
    std::vector<double> pomvec_nastav(poc_vstupu);
    for (int i = 0; i < rozmery[0]; ++i) {
        sit[0][i].set_vstupy (pomvec_nastav);
        sit[0][i].set_randomvahy();
        sit[0][i].vypocet();
        pom_vystup.push_back(sit[0][i].o);
    }
    pomvec_nastav.clear();

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

void NN::init_lstm(int poc_vstupu, const std::vector<int>& rozmers) {
    rozmery = rozmers;
    pocet_vrstev = rozmery.size();
    lstm_sit.clear();
    lstm_sit.resize(pocet_vrstev);
    for (int i = 0; i < pocet_vrstev; ++i) {
        lstm_sit[i].resize(rozmery[i]);
        for (int j = 0; j < rozmery[i]; ++j) {
            lstm_sit[i][j] = LSTMNeuron();
        }
    }
    pom_vystup.clear();
    vystupy.clear();
    std::vector<double> pomvec_nastav(poc_vstupu);
    for (int i = 0; i < rozmery[0]; ++i) {
        lstm_sit[0][i].set_vstupy(pomvec_nastav);
        lstm_sit[0][i].set_randomvahy();
        lstm_sit[0][i].vypocet();
        pom_vystup.push_back(lstm_sit[0][i].shortterm);
    }
    pomvec_nastav.clear();

    for (int i = 1; i < pocet_vrstev; ++i) {
        for (int j = 0; j < rozmery[i]; ++j) {
            lstm_sit[i][j].set_vstupy(pom_vystup);
            lstm_sit[i][j].set_randomvahy();
            lstm_sit[i][j].vypocet();
        }
        pom_vystup.clear();
        for (int j = 0; j < rozmery[i]; ++j) {
            pom_vystup.push_back( lstm_sit[i][j].shortterm);
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

void NN::online_bp_adam(int iter){
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
            for(int k = 0; k < sit[i][j].vahy.size();++k){
                sit[i][j].Mt[k] = beta*sit[i][j].Mt[k]+(1-beta)*(sit[i][j].delta * sit[i][j].vstupy[k]);
                sit[i][j].Vt[k] = beta2*sit[i][j].Vt[k]+(1-beta2)*pow((sit[i][j].delta * sit[i][j].vstupy[k]),2);
                sit[i][j].Mt_s[k] = sit[i][j].Mt[k]/(1-pow(beta,(m+1)));
                sit[i][j].Vt_s[k] = sit[i][j].Vt[k]/(1-pow(beta2,(m+1)));

                
                sit[i][j].vahy[k] = sit[i][j].vahy[k] - alfa * sit[i][j].Mt_s[k]/(sqrt(sit[i][j].Vt_s[k])+epsi);
        }
    }

                            
    }
}}}

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

void NN::online_lstm(int iter){
    for(int m = 0;m<iter;++m){
        std::cout<<m<<"\n";
    vystupy.clear();
    pom_vystup.clear();
    for (int l = 0;l<train_data.size();++l){
        for (int i = 0; i < rozmery[0]; ++i) {
            lstm_sit[0][i].set_vstupy (train_data[l]);
            lstm_sit[0][i].vypocet();
            pom_vystup.push_back(lstm_sit[0][i].shortterm);
        }
        
        for (int i = 1; i < pocet_vrstev; ++i) {
            for (int j = 0; j < rozmery[i]; ++j) {
                lstm_sit[i][j].set_vstupy(pom_vystup);
                lstm_sit[i][j].vypocet();
            }
            pom_vystup.clear();
            for (int j = 0; j < rozmery[i]; ++j) {
                pom_vystup.push_back(lstm_sit[i][j].shortterm);
            }
        }
        vystupy.push_back(pom_vystup[0]);
                            // dava smysl jen pro 1 vystup

//////////////////////////////////
//  BACKPROPAGACE 
//PŘEDĚLAT

    //     sit[pocet_vrstev-1][rozmery[pocet_vrstev-1]-1].delta = vystupy[l] - chtenejout[l];
    //     for (int i=0;i<rozmery[pocet_vrstev-2];++i){
    //         sit[pocet_vrstev-2][i].delta = sit[pocet_vrstev-2][i].der_akt_fun(sit[pocet_vrstev-2][i].a)*(sit[pocet_vrstev-1][rozmery[pocet_vrstev-1]-1].vahy[i] * sit[pocet_vrstev-1][rozmery[pocet_vrstev-1]-1].delta);
    //      }

    //     for(int j = (pocet_vrstev-3); j>=0;--j){
    //         for (int i=0;i<rozmery[j];++i){
    //              double skalsoucprv = 0.0;
    //             for(int k = 0;k<rozmery[j+1];++k){
    //                 skalsoucprv += sit[j+1][k].vahy[i] * sit[j+1][k].delta;
    //             }
            
    //         sit[j][i].delta = sit[j][i].der_akt_fun(sit[j][i].a)*skalsoucprv;
    //     }
    // }

    // for(int i = 0;i<pocet_vrstev;++i){
    //     for(int j = 0;j<rozmery[i];++j){
    //         for(int k = 0; k < sit[i][j].vahy.size();++k)
    //             sit[i][j].vahy[k] = sit[i][j].vahy[k] - alfa * sit[i][j].delta * sit[i][j].vstupy[k];
    //     }
    // }

                            
    }
}}


void NN::count_cost(){
    cost = 0.0;
    for(int i=0;i<vystupy.size();++i){
        cost += (vystupy[i]-chtenejout[i])*(vystupy[i]-chtenejout[i]);   
    }
    cost = cost/vystupy.size();
}

void NN::set_vstup_rada(const std::vector<double>& inputs) {
    vstupni_cr = inputs;
}


void NN::udelej_lag(size_t lag, bool tenzor){
    size_t nrows  = vstupni_cr.size() - lag+1;
    size_t ncols = lag;
    Matice<double> novej_lag(nrows,ncols);
    for (int i = 0; i<nrows;++i){
        for (int j = 0; j<ncols;++j){
            novej_lag.setElement(i,j,vstupni_cr[i+j]);
        }
    }

    if(tenzor == true){
        dataprocnn_t.add_matrix(novej_lag);
        kernely_t.add_matrix(set_kernel_rand(3, 3));
    } else{ 
        if(tenzor == false){
            dataprocnn_v2d.push_back(novej_lag);
            kernely_v2d.push_back( set_kernel_rand(3,3)); //zatim proste natvrdo 3
    }}
}

void NN::udelej_radky(size_t velrad, bool tenzor){
    int por = 0;
    size_t nrows  = vstupni_cr.size()/velrad;
    size_t ncols = velrad;
    Matice<double> novy_r(nrows,ncols);
    for (int i = 0; i<nrows;++i){
        for (int j = 0; j<ncols;++j){
            novy_r.setElement(i,j,vstupni_cr[por]);
            por+=1;
        }
    }
    if(tenzor == true){
        dataprocnn_t.add_matrix(novy_r);
        kernely_t.add_matrix(set_kernel_rand(3, 3));
    } else{ 
        if(tenzor == false){
            dataprocnn_v2d.push_back(novy_r);
            kernely_v2d.push_back( set_kernel_rand(3,3)); //zatim proste natvrdo 3
    }}
}

void NN::udelej_api(int n, double beta, Co coze, int kolik, bool tenzor){
    std::vector<double> api(vstupni_cr.size()-n+1);
    for(int i=0;i<api.size();++i){
        double appi = 0;
        int moc = 1;
        for(int j = n+i-1;j>(i-1);--j){
            appi += vstupni_cr[j]*pow(beta,moc);
            moc+=1;
        }
        api[i] = appi;
    }

    switch (coze){
        case radky:{
            int por = 0;
            size_t nrows  = api.size()/kolik;
            size_t ncols = kolik;
            Matice<double> novy_r(nrows,ncols);
            for (int i = 0; i<nrows;++i){
                for (int j = 0; j<ncols;++j){
                    novy_r.setElement(i,j,api[por]);
                    por+=1;
                }
             }
            if(tenzor == true){
                dataprocnn_t.add_matrix(novy_r);
            } else{ if (tenzor == false){
                dataprocnn_v2d.push_back(novy_r);
            }
            }
        break;}
        case lag:{
            size_t nrows  = api.size() - kolik+1;
            size_t ncols = kolik;
            Matice<double> novej_lag(nrows,ncols);
            for (int i = 0; i<nrows;++i){
                for (int j = 0; j<ncols;++j){
                    novej_lag.setElement(i,j,api[i+j]);
                }
            }
            if(tenzor == true){
                dataprocnn_t.add_matrix(novej_lag);
            } else{ if (tenzor == false){
                dataprocnn_v2d.push_back(novej_lag);
            }}
        break;
        }
    }
    if(tenzor == true){
        kernely_t.add_matrix( set_kernel_rand(3,3));
        } else{ if (tenzor == false){
    kernely_v2d.push_back( set_kernel_rand(3,3)); //zatim proste natvrdo 3
        }}
}


void NN::udelej_prumery(int n, Co coze, int kolik, bool tenzor){
    std::vector<double> prumery(vstupni_cr.size()-(2*n));
    for(int i=0;i<prumery.size();++i){
        double jprumer = 0;
        for(int j = 0;j<(2*n+1);++j){
            jprumer += vstupni_cr[j+i];
        }
        jprumer = jprumer/(2*n+1);
        prumery[i] = jprumer;
    }

    switch (coze){
        case radky:{
            int por = 0;
            size_t nrows  = prumery.size()/kolik;
            size_t ncols = kolik;
            Matice<double> novy_r(nrows,ncols);
            for (int i = 0; i<nrows;++i){
                for (int j = 0; j<ncols;++j){
                    novy_r.setElement(i,j,prumery[por]);
                    por+=1;
                }
             }
            if(tenzor == true){
                dataprocnn_t.add_matrix(novy_r);
            } else{ if (tenzor == false){
                dataprocnn_v2d.push_back(novy_r);
            }
            }
        break;}
        case lag:{
            size_t nrows  = prumery.size() - kolik+1;
            size_t ncols = kolik;
            Matice<double> novej_lag(nrows,ncols);
            for (int i = 0; i<nrows;++i){
                for (int j = 0; j<ncols;++j){
                    novej_lag.setElement(i,j,prumery[i+j]);
                }
            }
            if(tenzor == true){
                dataprocnn_t.add_matrix(novej_lag);
            } else{ if (tenzor == false){
                dataprocnn_v2d.push_back(novej_lag);
            }}
        break;
        }
    }
    if(tenzor == true){
        kernely_t.add_matrix( set_kernel_rand(3,3));
        } else{ if (tenzor == false){
    kernely_v2d.push_back( set_kernel_rand(3,3)); //zatim proste natvrdo 3
        }}
}



Matice<double> NN::set_kernel_rand(size_t radky,size_t sloupce){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.1, 0.9);
    size_t nrows  = radky;
    size_t ncols = sloupce;
    Matice<double> novej_k(nrows,ncols);
    for (int i = 0; i<nrows;++i){
        for (int j = 0; j<ncols;++j){
            novej_k.setElement(i,j,dis(gen));
        }
        }
    return novej_k;
}


void NN::cnn_pokus(int iter){
    Matice<double> jedna_kv((dataprocnn_v2d[0].getRows()-kernely_v2d[0].getRows()+1),(dataprocnn_v2d[0].getCols()-kernely_v2d[0].getCols()+1));

    for(int k = 0;k<(jedna_kv.getRows()-2);k++){                  // -2 taky zatim hardcode - event. zmenit na kernel size
        for(int l = 0;l<(jedna_kv.cols-2);l++){             
            double konvo = 0;
            for (int i = 0;i<kernely_v2d[0].getRows();i++){     //POUZIVEJ GETCOLS A GETROWS KDYZ UZ JE TAM MAS
                for(int j = 0;j<kernely_v2d[0].cols;j++){
                    konvo+=kernely_v2d[0](i,j)*dataprocnn_v2d[0](i+k,j+l);
                 }
            }
            jedna_kv.setElement(k,l,konvo);
        }
    }


    std::vector<double> max_kv(jedna_kv.cols);
    for(int i = 0;i<max_kv.size();i++){
        std::vector<double>pomocpls;
        pomocpls.clear();
        for(int j = 0;j<jedna_kv.getRows();j++){
            pomocpls.push_back(jedna_kv.getElement(j,i));
        }
        max_kv[i] = *std::max_element(pomocpls.begin(),pomocpls.end());
    }

for(int z = 0;z<iter;z++){
    vystupy.clear();
    pom_vystup.clear();
    for (int a = 0;a<(dataprocnn_v2d[0].getRows()-5);++a){
        for (int b = 0;b<(dataprocnn_v2d[0].cols-5);++b){
            std::vector<double>pomocvstupy = {dataprocnn_v2d[0](a,b),dataprocnn_v2d[0](a,b+1),dataprocnn_v2d[0](a,b+2),jedna_kv(a,b),max_kv[b]};
            //tohle zatim nějak hardcodnutý - nevim

            for (int i = 0; i < rozmery[0]; ++i) {
                sit[0][i].set_vstupy (pomocvstupy);
                sit[0][i].vypocet();
                pom_vystup.push_back(sit[0][i].o);
            }
        pomocvstupy.clear();
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

                                                                    //jak vlastne budou vypadat vystupy a co bude chtenejout?
        sit[pocet_vrstev-1][rozmery[pocet_vrstev-1]-1].delta = vystupy[a*dataprocnn_v2d[0].cols-1+b] - chtenejout[a*dataprocnn_v2d[0].cols-1+b];
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
            for(int k = 0; k < sit[i][j].vahy.size();++k){
                sit[i][j].Mt[k] = beta*sit[i][j].Mt[k]+(1-beta)*(sit[i][j].delta * sit[i][j].vstupy[k]);
                sit[i][j].Vt[k] = beta2*sit[i][j].Vt[k]+(1-beta2)*pow((sit[i][j].delta * sit[i][j].vstupy[k]),2);
                sit[i][j].Mt_s[k] = sit[i][j].Mt[k]/(1-pow(beta,(a*dataprocnn_v2d[0].cols+b+1)));
                sit[i][j].Vt_s[k] = sit[i][j].Vt[k]/(1-pow(beta2,(a*dataprocnn_v2d[0].cols+b+1)));

                
                sit[i][j].vahy[k] = sit[i][j].vahy[k] - alfa * sit[i][j].Mt_s[k]/(sqrt(sit[i][j].Vt_s[k])+epsi);
        }
    }

                            
    }}
}
std::cout<<z<<"\n";
}}

Matice<double> NN::max_pool(Matice<double> vstupnim, size_t oknorad, size_t oknosl){
    size_t radky = vstupnim.getRows()-oknorad+1;
    size_t sloupce = vstupnim.getCols()-oknosl+1;
    Matice<double>vystupm(radky,sloupce);
    double tedmax;
    for(int i = 0;i<radky;i++){
        for(int j = 0;j<sloupce;j++){
            for(int k = 0;k<oknorad;k++){
                for(int l = 0;l<oknosl;l++){
                    if((i+k)==0 and (j+l) == 0){
                        tedmax = vstupnim.getElement(0,0);
                    }else{
                        if(tedmax<vstupnim.getElement(i+k,j+l)){
                            tedmax = vstupnim.getElement(i+k,j+l);
                        }
                    }
                }
            }
            vystupm.setElement(i,j,tedmax);
        }
    }
    return vystupm;
}

Matice<double> NN::avg_pool(Matice<double> vstupnim, size_t oknorad, size_t oknosl){
    size_t radky = vstupnim.getRows()-oknorad+1;
    size_t sloupce = vstupnim.getCols()-oknosl+1;
    Matice<double>vystupm(radky,sloupce);
    for(int i = 0;i<radky;i++){
        for(int j = 0;j<sloupce;j++){
            double prumer = 0;
            for(int k = 0;k<oknorad;k++){
                for(int l = 0;l<oknosl;l++){
                    prumer+=vstupnim.getElement(i+k,j+l);
                    
                }
            }
            prumer = prumer/(oknorad + oknosl);
            vystupm.setElement(i,j,prumer);
        }
    }
    return vystupm;
}

Matice<double> NN::konvo(Matice<double> vstupnim, Matice<double> vstupkernel){
    size_t radky = vstupnim.getRows()-vstupkernel.getRows()+1;
    size_t sloupce = vstupnim.getCols()-vstupkernel.getCols()+1;
    Matice<double>vystupm(radky,sloupce);
    for(int i = 0;i<radky;i++){
        for(int j = 0;j<sloupce;j++){
            double konvol = 0;
            for(int k = 0;k<vstupkernel.getRows();k++){
                for(int l = 0;l<vstupkernel.getCols();l++){
                    konvol+=vstupnim.getElement(i+k,j+l)*vstupkernel.getElement(k,l);
                    
                }
            }
            vystupm.setElement(i,j,konvol);
        }
    }
    return vystupm;
}
