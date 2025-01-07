#include "neural_net.h"
#include <random>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <thread>

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
    val_data.clear();
    pom_vystup.clear();
    vystupy.clear();
    chtenejout.clear();
}

NN::~NN()   // Destruktor
{
    rozmery.clear();
    sit.clear();
    train_data.clear();
    val_data.clear();
    pom_vystup.clear();
    vystupy.clear();
    chtenejout.clear();
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
    val_data(other.val_data),
    pom_vystup(other.pom_vystup),
    vystupy(other.vystupy),
    chtenejout(other.chtenejout){}

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
    val_data(std::move(other.val_data)),
    pom_vystup(std::move(other.pom_vystup)),
    vystupy(std::move(other.vystupy)),
    chtenejout(std::move(other.chtenejout)){

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
    other.val_data.clear();
    other.pom_vystup.clear();
    other.vystupy.clear();
    other.chtenejout.clear();
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
        val_data = other.val_data;
        pom_vystup = other.pom_vystup;
        vystupy = other.vystupy;
        chtenejout = other.chtenejout;
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
        val_data = std::move(other.val_data);
        pom_vystup = std::move(other.pom_vystup);
        vystupy = std::move(other.vystupy);
        chtenejout = std::move(other.chtenejout);

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
        other.val_data.clear();
        other.pom_vystup.clear();
        other.vystupy.clear();
        other.chtenejout.clear();
    }
    return *this;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////ú
double NN::tanh(double x){
    return (exp(x)-exp(-x))/(exp(x)+exp(-x));
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
    std::cout<<sit[pocet_vrstev-1][rozmery[pocet_vrstev-1]-1].delta;
}

}

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
}} std::cout<<chtenejout[0]<<"   "<<vystupy[0];}

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
    cost = cost/(2*vystupy.size());
}

void NN::print_vystup(){
    for (int i = 0;i<vystupy.size();++i){
        std::cout<<vystupy[i]<<" ";
    }
}

// void NN::set_vstup_rada(const std::vector<double>& inputs) {
//     vstupni_cr = inputs;
// }

Matice<double> NN::udelej_lag(size_t lag,const std::vector<double>& cr){
    size_t nrows  = cr.size() - lag+1;
    size_t ncols = lag;
    Matice<double> novej_lag(nrows,ncols);
    for (int i = 0; i<nrows;++i){
        for (int j = 0; j<ncols;++j){
            novej_lag.setElement(i,j,cr[i+j]);
        }
    }
return novej_lag;
}

Matice<double> NN::udelej_radky(size_t velrad,const std::vector<double>& cr){
    int por = 0;
    size_t nrows  = cr.size()/velrad;
    size_t ncols = velrad;
    Matice<double> novy_r(nrows,ncols);
    for (int i = 0; i<nrows;++i){
        for (int j = 0; j<ncols;++j){
            novy_r.setElement(i,j,cr[por]);
            por+=1;
        }
    }
return novy_r;
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

Tenzor<double> NN::konvo_3d(Tenzor<double> vstupnim, Tenzor<double> vstupkernel){
    size_t radky = vstupnim.getRows()-vstupkernel.getRows()+1;
    size_t sloupce = vstupnim.getCols()-vstupkernel.getCols()+1;
    size_t vrstvy = vstupnim.getDepth()*vstupkernel.getDepth();
    Tenzor<double>vystupm(vrstvy,radky,sloupce);

    for(int tv = 0;tv<vstupnim.getDepth();++tv){
        for(int krs = 0;krs<vstupkernel.getDepth();++krs){
            for(int i = 0;i<radky;i++){
                for(int j = 0;j<sloupce;j++){
                    double konvol = 0;
                    for(int k = 0;k<vstupkernel.getRows();k++){
                        for(int l = 0;l<vstupkernel.getCols();l++){
                            konvol+=vstupnim.getElement(tv,i+k,j+l)*vstupkernel.getElement(krs,k,l);
                        }
                    }
                    vystupm.setElement((krs*vstupnim.getDepth()+tv),i,j,konvol);
                }
            }
        }
    }
    return vystupm;
}

Matice<double> NN::konvo_fullstep(Matice<double> vstupnim, Matice<double> vstupkernel){
    size_t radky = vstupnim.getRows()/vstupkernel.getRows();
    size_t sloupce = vstupnim.getCols()/vstupkernel.getCols();
    Matice<double>vystupm(radky,sloupce);
    double konvol;
    for(int i = 0;i<radky;i++){
        for(int j = 0;j<sloupce;j++){
            konvol = 0;
            for(int k = 0;k<vstupkernel.getRows();k++){
                for(int l = 0;l<vstupkernel.getCols();l++){
                    konvol+=vstupnim.getElement(i*vstupkernel.getRows()+k,j*vstupkernel.getCols()+l)*vstupkernel.getElement(k,l);
                    
                }
            }
            vystupm.setElement(i,j,konvol);
        }
    }
    return vystupm;
}

Tenzor<double> NN::konvo_fullstep_3d(Tenzor<double> vstupnt, Tenzor<double> vstupker){
    size_t radky = vstupnt.getRows()/vstupker.getRows();
    size_t sloupce = vstupnt.getCols()/vstupker.getCols();
    size_t vrstvy = vstupnt.getDepth()*vstupker.getDepth();
    Tenzor<double>vystupt(vrstvy,radky,sloupce);
    double konvol;

    for(int krs = 0;krs<vstupker.getDepth();++krs){
        for(int tv = 0;tv<vstupnt.getDepth();++tv){
            for(int i = 0;i<radky;i++){
                for(int j = 0;j<sloupce;j++){
                    konvol = 0;
                    for(int k = 0;k<vstupker.getRows();k++){
                        for(int l = 0;l<vstupker.getCols();l++){
                            konvol+=vstupnt.getElement(tv,i*vstupker.getRows()+k,j*vstupker.getCols()+l)*vstupker.getElement(krs,k,l);
                        }
                    }
                    vystupt.setElement((krs*vstupnt.getDepth()+tv),i,j,konvol);
                }
            }
        }
    }
    return vystupt;
}

Tenzor<double> NN::konvo_fullstep_3d_1by1(Tenzor<double> vstupnt, Tenzor<double> vstupker){
    if (vstupnt.getDepth() != vstupker.getDepth()) {
        std::cout << "tenzor a kernely nemají stejně vrstev";
        exit(0);
    }
    size_t radky = vstupnt.getRows()/vstupker.getRows();
    size_t sloupce = vstupnt.getCols()/vstupker.getCols();
    size_t vrstvy = vstupnt.getDepth();
    Tenzor<double>vystupt(vrstvy,radky,sloupce);
    double konvol;


        for(int tv = 0;tv<vstupnt.getDepth();++tv){
            for(int i = 0;i<radky;i++){
                for(int j = 0;j<sloupce;j++){
                    konvol = 0;
                    for(int k = 0;k<vstupker.getRows();k++){
                        for(int l = 0;l<vstupker.getCols();l++){
                            konvol+=vstupnt.getElement(tv,i*vstupker.getRows()+k,j*vstupker.getCols()+l)*vstupker.getElement(tv,k,l);
                        }
                    }
                    vystupt.setElement(tv,i,j,konvol);
                }
            }
        }
    return vystupt;
}

void NN::cnnonfly_cal(size_t vel_ker, size_t poc_ker, int iter){
    int rok = 365;
    dataprocnn = udelej_radky(rok,Q_kal_vstup);
    dataprocnn.sloupce_nakonec(vel_ker - 1);
    kernely_onfly.resize(poc_ker,vel_ker,vel_ker);
    kernely_onfly.rand_vypln(0.0,0.1);

// for(int i = 0; i<kernely_onfly.getDepth();i++){
//    for(int j = 0; j<kernely_onfly.getRows();j++){
//     for(int k = 0; k<kernely_onfly.getCols();k++){
//         kernely_onfly.setElement(i,j,k,(0.1+k*0.01+j*0.05));
// }
// } 
// }

    Tenzor<double> akt_vstup(1,vel_ker,vel_ker);
    Tenzor<double> deltazmlp(1,1,1);
    Tenzor<double> uprava_k(poc_ker,vel_ker,vel_ker);
    Tenzor<double> vrstva_vystup;
    std::vector<double> vystzkonv;
    
    Matice<double> biasmatice(1,poc_ker);
    biasmatice.rand_vypln(0.0,0.1);    
    std::vector<double> biaskonv;
    for(int i = 0; i<poc_ker;i++){
        biaskonv.push_back(biasmatice.getElement(0,i));
    }


//////////////////////////////////////////////////////////// KALIBRACE //////////////////////////////////////////////////////
    for(int m = 0; m < iter; m++){
        std::cout<<m<<"\n";
        for(int roky = 0; roky < (dataprocnn.getRows() - (vel_ker - 1)); roky++){
            for(int dny = 0; dny < rok ; dny++){

//// KONVOLUCE //////////////////////////////////////////////////////////////////////////////////////////////////////////////                
                for(int i = 0; i < vel_ker; i++){
                    for(int j = 0; j < vel_ker; j++){
                        akt_vstup.setElement(0,i,j,dataprocnn.getElement(i+roky,j+dny));
                    }
                }

                vrstva_vystup = konvo_3d(akt_vstup,kernely_onfly);

                for (int i = 0; i<vrstva_vystup.getDepth();i++){
                    vrstva_vystup.setElement(i,0,0,(vrstva_vystup.getElement(i,0,0) + biaskonv[i]));
                }

                vystzkonv.clear();
                for(int i = 0;i<vrstva_vystup.getDepth();++i){
                    if(vrstva_vystup.getElement(i,0,0) < 0.0){
                        vystzkonv.push_back(0.01 * vrstva_vystup.getElement(i,0,0));
                        } else {
                            vystzkonv.push_back(vrstva_vystup.getElement(i,0,0));
                        }
                        
                }

//// MLP //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                pom_vystup.clear();
                for (int i = 0; i < rozmery[0]; ++i) {
                    sit[0][i].set_vstupy(vystzkonv);
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

//// MLP BACKPROP ////////////////////////////////////////////////////////////////////////////////////////////////////////////
                sit[pocet_vrstev-1][rozmery[pocet_vrstev-1]-1].delta = pom_vystup[0] - chtenejout[roky*rok+dny];

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

    // for(int i = 0;i<pocet_vrstev;++i){
    //     for(int j = 0;j<rozmery[i];++j){
    //         for(int k = 0; k < sit[i][j].vahy.size();++k){
    //             sit[i][j].Mt[k] = beta*sit[i][j].Mt[k]+(1-beta)*(sit[i][j].delta * sit[i][j].vstupy[k]);
    //             sit[i][j].Vt[k] = beta2*sit[i][j].Vt[k]+(1-beta2)*pow((sit[i][j].delta * sit[i][j].vstupy[k]),2);
    //             sit[i][j].Mt_s[k] = sit[i][j].Mt[k]/(1-pow(beta,(m+1)));
    //             sit[i][j].Vt_s[k] = sit[i][j].Vt[k]/(1-pow(beta2,(m+1)));

                
    //             sit[i][j].vahy[k] = sit[i][j].vahy[k] - alfa * sit[i][j].Mt_s[k]/(sqrt(sit[i][j].Vt_s[k])+epsi);
    //     }
    // }                     
    // }

//// CNN BACKPROP ////////////////////////////////////////////////////////////////////////////////////////////////////////////
                for (int neur = 0;neur<rozmery[0];++neur){
                    deltazmlp.setElement(0,0,0,sit[0][neur].delta);
                    uprava_k = konvo_3d(akt_vstup,deltazmlp); 
                    for(int upr = 0;upr<kernely_onfly.getDepth();++upr){
                        for(int sl = 0; sl<kernely_onfly.getCols();++sl){
                            for(int rad = 0;rad<kernely_onfly.getRows();++rad){
                                
                                if(vystzkonv[upr] < 0.0){
                                    kernely_onfly.setElement(upr,rad,sl,kernely_onfly.getElement(upr,rad,sl)-alfa*0.01*uprava_k.getElement(0,rad,sl));
                                } else{
                                    kernely_onfly.setElement(upr,rad,sl,kernely_onfly.getElement(upr,rad,sl)-alfa*uprava_k.getElement(0,rad,sl));
                                }
                            }
                        }
                        if(vystzkonv[upr] < 0.0){
                            biaskonv[upr] = biaskonv[upr] - alfa*0.01*sit[0][neur].delta;
                        } else {
                            biaskonv[upr] = biaskonv[upr] - alfa*sit[0][neur].delta;
                        }
                    }
                }
            }
        }
    }

///////////////////////////////////////////////////////////////// VYPOCET ////////////////////////////////////////////////
    vystupy.clear();
    for(int roky = 0; roky < (dataprocnn.getRows() - (vel_ker - 1)); roky++){
            for(int dny = 0; dny < rok ; dny++){

//// KONVOLUCE //////////////////////////////////////////////////////////////////////////////////////////////////////////////                
                for(int i = 0; i < vel_ker; i++){
                    for(int j = 0; j < vel_ker; j++){
                        akt_vstup.setElement(0,i,j,dataprocnn.getElement(i+roky,j+dny));
                    }
                }

                vrstva_vystup = konvo_3d(akt_vstup,kernely_onfly);

                for (int i = 0; i<vrstva_vystup.getDepth();i++){
                    vrstva_vystup.setElement(i,0,0,(vrstva_vystup.getElement(i,0,0) + biaskonv[i]));
                }

                vystzkonv.clear();
                for(int i = 0;i<vrstva_vystup.getDepth();++i){
                    if(vrstva_vystup.getElement(i,0,0) < 0.0){
                        vystzkonv.push_back(0.01 * vrstva_vystup.getElement(i,0,0));
                        } else {
                            vystzkonv.push_back(vrstva_vystup.getElement(i,0,0));
                        }
                        
                }

//// MLP //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                pom_vystup.clear();
                for (int i = 0; i < rozmery[0]; ++i) {
                    sit[0][i].set_vstupy (vystzkonv);
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
            }           
        }
    }

void NN::cnnonfly_val(){
    int rok = 365;
    int vel_ker = kernely_onfly.getRows();
    dataprocnn = udelej_radky(rok,Q_val_vstup);
    dataprocnn.sloupce_nakonec(vel_ker - 1);
    Tenzor<double> akt_vstup(1,vel_ker,vel_ker);
    Tenzor<double> vrstva_vystup;
    std::vector<double> vystzkonv;

    vystupy.clear();
    for(int roky = 0; roky < (dataprocnn.getRows() - (vel_ker - 1)); roky++){
            for(int dny = 0; dny < rok ; dny++){

//// KONVOLUCE //////////////////////////////////////////////////////////////////////////////////////////////////////////////                
                for(int i = 0; i < vel_ker; i++){
                    for(int j = 0; j < vel_ker; j++){
                        akt_vstup.setElement(0,i,j,dataprocnn.getElement(i+roky,j+dny));
                    }
                }

                vrstva_vystup = konvo_3d(akt_vstup,kernely_onfly);
                vystzkonv.clear();
                for(int i = 0;i<vrstva_vystup.getDepth();++i){
                    if(vrstva_vystup.getElement(i,0,0) < 0.0){
                        vystzkonv.push_back(0.01 * vrstva_vystup.getElement(i,0,0));
                        } else {
                            vystzkonv.push_back(vrstva_vystup.getElement(i,0,0));
                        }
                        
                }

//// MLP //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                pom_vystup.clear();
                for (int i = 0; i < rozmery[0]; ++i) {
                    sit[0][i].set_vstupy (vystzkonv);
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
            }           
        }
}

void NN::cnn_full_cal(int iter){
    alfa = 0.01;

    kernely_full_1.resize(5,6,6);
    kernely_full_1.rand_vypln(0.0,0.1);
    kernely_full_2s.resize(5,3,3);
    kernely_full_2s.rand_vypln(0.0,0.1);
    kernely_full_3.resize(10,3,3);
    kernely_full_3.rand_vypln(0.0,0.1);
    kernely_full_4s.resize(50,3,3);
    kernely_full_4s.rand_vypln(0.0,0.1);

    ////////////////////////////////KONVOLUCE
    Tenzor<double> vrstva0;
    Tenzor<double> vrstva1;
    Tenzor<double> vrstva2;
    Tenzor<double> vrstva3;
    Tenzor<double> vrstva_final;
    Tenzor<double> deltazmlp(1,1,1);
    Tenzor<double> grad_3_4;
    Tenzor<double> uprava_k4;
    Tenzor<double> uprava_k3;
    Tenzor<double> grad_2_3;
    Tenzor<double> grad_1_2;
    Tenzor<double> uprava_k2;
    Tenzor<double> uprava_k1;
    std::vector<double> vystzkonv;
    std::vector<double> current_kus;

for(int ite = 0;ite<iter;++ite){
    for (int kroky = 0; kroky < (Q_kal_vstup.size() - 420); kroky++){
        
        vrstva0.resize(0,0,0);
        vrstva1.resize(0,0,0);
        vrstva2.resize(0,0,0);
        vrstva3.resize(0,0,0);
        vrstva_final.resize(0,0,0);

        for(int kus = 0; kus < 400; kus++){
            current_kus.push_back(Q_kal_vstup[kroky + kus]);
        }
        chtenejout.clear();
        for(int cht = 0;cht< 20;cht++){
            chtenejout.push_back(Q_kal_vstup[400+cht+kroky]);
        }

        dataprocnn = udelej_radky(20,current_kus);
        current_kus.clear();
        vrstva0.add_matrix(dataprocnn);
        dataprocnn.resize(0,0);
        vrstva1 = konvo_3d(vrstva0,kernely_full_1);
        vrstva2 = konvo_fullstep_3d_1by1(vrstva1,kernely_full_2s);
        vrstva3 = konvo_3d(vrstva2,kernely_full_3);
        vrstva_final = konvo_fullstep_3d_1by1(vrstva3,kernely_full_4s);

    ///////////////////////////MLP

    vystupy.clear();
    vystzkonv.clear();
    for(int i = 0;i<vrstva_final.getDepth();++i){
        vystzkonv.push_back(vrstva_final.getElement(i,0,0));
    }
        
        pom_vystup.clear();
            for (int i = 0; i < rozmery[0]; ++i) {
                sit[0][i].set_vstupy (vystzkonv);
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
            vystupy = pom_vystup;

///////////// MLP BACKPROP
        count_cost();
        sit[pocet_vrstev-1][rozmery[pocet_vrstev-1]-1].delta = cost;
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

    ///////////// KONVO BACKPROP
    for (int neur = 0;neur<rozmery[0];++neur){
    deltazmlp.setElement(0,0,0,sit[0][neur].delta);

    grad_3_4.resize(0,0,0);
    grad_3_4 = kernely_full_4s;
    grad_3_4.flip180();
    grad_3_4 = konvo_3d(grad_3_4,deltazmlp);

    uprava_k4 = konvo_3d(vrstva3,deltazmlp);


        for(int upr = 0;upr<uprava_k3.getDepth();++upr){
            for(int sl = 0; sl<kernely_full_4s.getCols();++sl){
                for(int rad = 0;rad<kernely_full_4s.getRows();++rad){
                    kernely_full_4s.setElement(upr,rad,sl,kernely_full_4s.getElement(upr,rad,sl)-alfa*uprava_k3.getElement(upr,rad,sl));
                }
            }
        }
        uprava_k4.resize(0,0,0);

    grad_2_3.resize(0,0,0);
    grad_2_3 = kernely_full_3;
    grad_2_3.flip180();

    uprava_k3 = konvo_3d(vrstva2,grad_3_4);

    grad_3_4.obal_nul(2);
    grad_2_3 = konvo_3d(grad_3_4,grad_2_3);



    for(int ker = 0;ker<kernely_full_3.getDepth();++ker){
        for(int upr = 0;upr<uprava_k3.getDepth();++upr){
            for(int sl = 0; sl<kernely_full_3.getCols();++sl){
                for(int rad = 0;rad<kernely_full_3.getRows();++rad){
                    kernely_full_3.setElement(ker,rad,sl,kernely_full_3.getElement(ker,rad,sl)-alfa*uprava_k2.getElement(upr,rad,sl));
                }
            }
        }
    }
    uprava_k3.resize(0,0,0);

    grad_1_2.resize(0,0,0);
    grad_1_2 = kernely_full_2s;
    grad_1_2.flip180();
    grad_2_3.dilace(2,2);

    uprava_k2 = konvo_3d(vrstva1,grad_2_3);

    grad_2_3.obal_nul(2);
    grad_1_2 = konvo_3d(grad_2_3,grad_1_2);



    for(int ker = 0;ker<kernely_full_2s.getDepth();++ker){
        for(int upr = 0;upr<500;++upr){
            for(int sl = 0; sl<kernely_full_2s.getCols();++sl){
                for(int rad = 0;rad<kernely_full_2s.getRows();++rad){
                    kernely_full_2s.setElement(ker,rad,sl,kernely_full_2s.getElement(ker,rad,sl)-alfa*uprava_k2.getElement((ker*500+upr),rad,sl));
                }
            }
        }
    }
    uprava_k2.resize(0,0,0);

    uprava_k1 = konvo_3d(vrstva0,grad_1_2);



    for(int ker = 0;ker<kernely_full_1.getDepth();++ker){
        for(int upr = 0;upr<uprava_k1.getDepth();++upr){
            for(int sl = 0; sl<kernely_full_1.getCols();++sl){
                for(int rad = 0;rad<kernely_full_1.getRows();++rad){
                    kernely_full_1.setElement(ker,rad,sl,kernely_full_1.getElement(ker,rad,sl)-alfa*uprava_k1.getElement(upr,rad,sl));
                }
            }
        }
    }
    uprava_k1.resize(0,0,0);
    }
}
}
}

void NN::cnn_full_val(){
    
    if (Q_val_vstup.size() < 400 || Q_val_vstup.size() % 20 != 0) {
        std::cout << "Delka Q_val_vstup musi byt vetsi nez 400 a byt delitelna 20";
        exit(0);
    }

    ////////////////////////////////KONVOLUCE
    Tenzor<double> vrstva0;
    Tenzor<double> vrstva1;
    Tenzor<double> vrstva2;
    Tenzor<double> vrstva3;
    Tenzor<double> vrstva_final;
    std::vector<double> vystzkonv;
    std::vector<double> current_kus;
    vystupy.clear();

    for (int kroky = 0; kroky < ((Q_val_vstup.size()-400)/20);kroky++){
        vrstva0.resize(0,0,0);
        vrstva1.resize(0,0,0);
        vrstva2.resize(0,0,0);
        vrstva3.resize(0,0,0);
        vrstva_final.resize(0,0,0);

        for(int kus = 0; kus < 400; kus++){
            current_kus.push_back(Q_val_vstup[kroky*20+kus]);
        }

        dataprocnn = udelej_radky(20,current_kus);
        current_kus.clear();
        vrstva0.add_matrix(dataprocnn);
        dataprocnn.resize(0,0);
        vrstva1 = konvo_3d(vrstva0,kernely_full_1);
        vrstva2 = konvo_fullstep_3d_1by1(vrstva1,kernely_full_2s);
        vrstva3 = konvo_3d(vrstva2,kernely_full_3);
        vrstva_final = konvo_fullstep_3d_1by1(vrstva3,kernely_full_4s);

    ///////////////////////////MLP

    vystzkonv.clear();
    for(int i = 0;i<vrstva_final.getDepth();++i){
        vystzkonv.push_back(vrstva_final.getElement(i,0,0));
    }
        
        pom_vystup.clear();
            for (int i = 0; i < rozmery[0]; ++i) {
                sit[0][i].set_vstupy (vystzkonv);
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
            for(int vys = 0; vys<pom_vystup.size();vys++){
                vystupy.push_back(pom_vystup[vys]);
            }
    }
}

void NN::cnn1D_cal(size_t vel_ker, size_t poc_ker, int iter){
    
    kernely_1D.resize(3,poc_ker,vel_ker);
    kernely_1D.rand_vypln(0.0,0.1);
    double deltazmlp;
    Tenzor<double> uprava_k(3,poc_ker,vel_ker);
    std::vector<double> vystzkonv;   
    Matice<double> biaskonv(3,poc_ker);
    
    if (Q_kal_vstup.size() != R_kal_vstup.size()|| Q_kal_vstup.size() != T_kal_vstup.size()) {
        std::cout << "vstupni řady nejsou stejně dlouhý";
        exit(0);
    }

//////////////////////////////////////////////////////////// KALIBRACE //////////////////////////////////////////////////////
    for(int m = 0; m < iter; m++){
        std::cout<<m<<"\n";
        for(int kroky = 0; kroky < (Q_kal_vstup.size() - vel_ker); kroky++){

//// KONVOLUCE //////////////////////////////////////////////////////////////////////////////////////////////////////////////                
                vystzkonv.clear();

                for(int i = 0; i < poc_ker; i++){
                    double konvo = 0.0;
                    for(int j = 0; j < vel_ker; j++){
                        konvo += Q_kal_vstup[kroky+j] * kernely_1D.getElement(0,i,j);
                    }
                    konvo += biaskonv.getElement(0,i);
                    if(konvo < 0.0){
                        vystzkonv.push_back(konvo*0.01);
                    }else{
                        vystzkonv.push_back(konvo);
                    }
                }

                for(int i = 0; i < poc_ker; i++){
                    double konvo = 0.0;
                    for(int j = 0; j < vel_ker; j++){
                        konvo += R_kal_vstup[kroky+j] * kernely_1D.getElement(1,i,j);
                    }
                    konvo += biaskonv.getElement(1,i);
                    if(konvo < 0.0){
                        vystzkonv.push_back(konvo*0.01);
                    }else{
                        vystzkonv.push_back(konvo);
                    }
                }

                for(int i = 0; i < poc_ker; i++){
                    double konvo = 0.0;
                    for(int j = 0; j < vel_ker; j++){
                        konvo += T_kal_vstup[kroky+j] * kernely_1D.getElement(2,i,j);
                    }
                    konvo += biaskonv.getElement(2,i);
                    if(konvo < 0.0){
                        vystzkonv.push_back(konvo*0.01);
                    }else{
                        vystzkonv.push_back(konvo);
                    }
                }

//// MLP //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                pom_vystup.clear();
                for (int i = 0; i < rozmery[0]; ++i) {
                    sit[0][i].set_vstupy(vystzkonv);
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

//// MLP BACKPROP ////////////////////////////////////////////////////////////////////////////////////////////////////////////
                sit[pocet_vrstev-1][rozmery[pocet_vrstev-1]-1].delta = pom_vystup[0] - Q_kal_vstup[kroky+vel_ker];

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

//// CNN BACKPROP ////////////////////////////////////////////////////////////////////////////////////////////////////////////
                for (int neur = 0;neur<rozmery[0];++neur){
                    deltazmlp = sit[0][neur].delta;

                    for(int i = 0; i < poc_ker; i++){
                        for(int j = 0; j < vel_ker; j++){
                            uprava_k.setElement(0,i,j,(deltazmlp*Q_kal_vstup[kroky+j]));
                        }
                    }

                    for(int i = 0; i < poc_ker; i++){
                        for(int j = 0; j < vel_ker; j++){
                            uprava_k.setElement(1,i,j,(deltazmlp*R_kal_vstup[kroky+j]));
                        }
                    }

                    for(int i = 0; i < poc_ker; i++){
                        for(int j = 0; j < vel_ker; j++){
                            uprava_k.setElement(2,i,j,(deltazmlp*T_kal_vstup[kroky+j]));
                        }
                    }

                    for(int i = 0; i < 3; i++){
                        for(int j = 0; j < poc_ker; j++){
                            if(vystzkonv[i*poc_ker+j] > 0.0){
                                for(int k = 0; k < vel_ker; k++){
                                    kernely_1D.setElement(i,j,k,(kernely_1D.getElement(i,j,k) - alfa * uprava_k.getElement(i,j,k)));
                                }
                                biaskonv.setElement(i,j,(biaskonv.getElement(i,j) - alfa * deltazmlp));
                            } else{
                                for(int k = 0; k < vel_ker; k++){
                                    kernely_1D.setElement(i,j,k,(kernely_1D.getElement(i,j,k) - alfa * 0.01 * uprava_k.getElement(i,j,k)));
                                }
                                biaskonv.setElement(i,j,(biaskonv.getElement(i,j) - alfa * 0.01 * deltazmlp));
                            }

                        }
                    }
                }
        }
    }

///////////////////////////////////////////////////////////////// VYPOCET ////////////////////////////////////////////////
    vystupy.clear();
    for(int kroky = 0; kroky < (Q_kal_vstup.size() - vel_ker); kroky++){

//// KONVOLUCE //////////////////////////////////////////////////////////////////////////////////////////////////////////////                
                vystzkonv.clear();

                for(int i = 0; i < poc_ker; i++){
                    double konvo = 0.0;
                    for(int j = 0; j < vel_ker; j++){
                        konvo += Q_kal_vstup[kroky+j] * kernely_1D.getElement(0,i,j);
                    }
                    konvo += biaskonv.getElement(0,i);
                    if(konvo < 0.0){
                        vystzkonv.push_back(konvo*0.01);
                    }else{
                        vystzkonv.push_back(konvo);
                    }
                }

                for(int i = 0; i < poc_ker; i++){
                    double konvo = 0.0;
                    for(int j = 0; j < vel_ker; j++){
                        konvo += R_kal_vstup[kroky+j] * kernely_1D.getElement(1,i,j);
                    }
                    konvo += biaskonv.getElement(1,i);
                    if(konvo < 0.0){
                        vystzkonv.push_back(konvo*0.01);
                    }else{
                        vystzkonv.push_back(konvo);
                    }
                }

                for(int i = 0; i < poc_ker; i++){
                    double konvo = 0.0;
                    for(int j = 0; j < vel_ker; j++){
                        konvo += T_kal_vstup[kroky+j] * kernely_1D.getElement(2,i,j);
                    }
                    konvo += biaskonv.getElement(2,i);
                    if(konvo < 0.0){
                        vystzkonv.push_back(konvo*0.01);
                    }else{
                        vystzkonv.push_back(konvo);
                    }
                }

//// MLP //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                pom_vystup.clear();
                for (int i = 0; i < rozmery[0]; ++i) {
                    sit[0][i].set_vstupy(vystzkonv);
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
            }           
        }

void NN::cnn1D_val(){
    std::vector<double> vystzkonv;   
    Matice<double> biaskonv(3,kernely_1D.getRows());
    
    if (Q_val_vstup.size() != R_val_vstup.size()|| Q_val_vstup.size() != T_val_vstup.size()) {
        std::cout << "vstupni řady nejsou stejně dlouhý";
        exit(0);
    }

    vystupy.clear();
    for(int kroky = 0; kroky < (Q_val_vstup.size() - kernely_1D.getCols()); kroky++){

//// KONVOLUCE //////////////////////////////////////////////////////////////////////////////////////////////////////////////                
                vystzkonv.clear();

                for(int i = 0; i < kernely_1D.getRows(); i++){
                    double konvo = 0.0;
                    for(int j = 0; j < kernely_1D.getCols(); j++){
                        konvo += Q_val_vstup[kroky+j] * kernely_1D.getElement(0,i,j);
                    }
                    konvo += biaskonv.getElement(0,i);
                    if(konvo < 0.0){
                        vystzkonv.push_back(konvo*0.01);
                    }else{
                        vystzkonv.push_back(konvo);
                    }
                }

                for(int i = 0; i < kernely_1D.getRows(); i++){
                    double konvo = 0.0;
                    for(int j = 0; j < kernely_1D.getCols(); j++){
                        konvo += R_val_vstup[kroky+j] * kernely_1D.getElement(1,i,j);
                    }
                    konvo += biaskonv.getElement(1,i);
                    if(konvo < 0.0){
                        vystzkonv.push_back(konvo*0.01);
                    }else{
                        vystzkonv.push_back(konvo);
                    }
                }

                for(int i = 0; i < kernely_1D.getRows(); i++){
                    double konvo = 0.0;
                    for(int j = 0; j < kernely_1D.getCols(); j++){
                        konvo += T_val_vstup[kroky+j] * kernely_1D.getElement(2,i,j);
                    }
                    konvo += biaskonv.getElement(2,i);
                    if(konvo < 0.0){
                        vystzkonv.push_back(konvo*0.01);
                    }else{
                        vystzkonv.push_back(konvo);
                    }
                }

//// MLP //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                pom_vystup.clear();
                for (int i = 0; i < rozmery[0]; ++i) {
                    sit[0][i].set_vstupy(vystzkonv);
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
            }   
}