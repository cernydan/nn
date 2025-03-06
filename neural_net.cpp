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

void NN::set_vstup_rady(const std::vector<double>& Qkal_in, const std::vector<double>& Qval_in,
                        const std::vector<double>& Rkal_in, const std::vector<double>& Rval_in,
                        const std::vector<double>& Tkal_in, const std::vector<double>& Tval_in) {
    Q_kal_vstup = Qkal_in;
    Q_val_vstup = Qval_in;
    R_kal_vstup = Rkal_in;
    R_val_vstup = Rval_in;
    T_kal_vstup = Tkal_in;
    T_val_vstup = Tval_in;
}

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

Tenzor<double> NN::max_pool_fullstep_3d(Tenzor<double>& vstupnim, size_t oknorad, size_t oknosl){
    size_t radky = vstupnim.getRows()/oknorad;
    size_t sloupce = vstupnim.getCols()/oknosl;
    size_t vrstvy = vstupnim.getDepth();
    Tenzor<double>vystupm(vrstvy,radky,sloupce);
    double tedmax;
    for (int hl = 0;hl<vrstvy;++hl){
        for(int i = 0;i<radky;i++){
            for(int j = 0;j<sloupce;j++){
                for(int k = 0;k<oknorad;k++){
                    for(int l = 0;l<oknosl;l++){
                        if(k==0 and l == 0){
                            tedmax = vstupnim.getElement(hl,i*oknorad,j*oknosl);
                        }else{
                            if(tedmax<vstupnim.getElement(hl,i*oknorad+k,j*oknosl+l)){
                                tedmax = vstupnim.getElement(hl,i*oknorad+k,j*oknosl+l);
                            }
                        }
                    }
                }
                vystupm.setElement(hl,i,j,tedmax);
            }
        }
    }
    return vystupm;
}


Matice<double> NN::konvo(Matice<double>& vstupnim, Matice<double>& vstupkernel){
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

Tenzor<double> NN::konvo_3d(Tenzor<double>& vstupnim, Tenzor<double>& vstupkernel){
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
                    vystupm.setElement((tv*vstupkernel.getDepth()+krs),i,j,konvol);
                }
            }
        }
    }
    return vystupm;
}

Matice<double> NN::konvo_fullstep(Matice<double>& vstupnim, Matice<double>& vstupkernel){
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

Tenzor<double> NN::konvo_fullstep_3d(Tenzor<double>& vstupnt, Tenzor<double>& vstupker){
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

Tenzor<double> NN::konvo_fullstep_3d_1by1(Tenzor<double>& vstupnt, Tenzor<double>& vstupker){
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

void NN::cnnonfly_cal(size_t row_ker,size_t col_ker, size_t poc_ker, int iter, int velic){
    int rok = 365;
    if (velic == 2){
    Matice<double> dataprocnn_Q;
    Matice<double> dataprocnn_R;
    dataprocnn_Q = udelej_radky(rok,Q_kal_vstup);
    dataprocnn_Q.sloupce_nakonec(col_ker - 1);
    dataprocnn_R = udelej_radky(rok,R_kal_vstup);
    dataprocnn_R.sloupce_nakonec(col_ker - 1);
    kernely_onfly_Q.resize(poc_ker,row_ker,col_ker);
    kernely_onfly_Q.rand_vypln(0.0,0.1);
    kernely_onfly_R.resize(poc_ker,row_ker,col_ker);
    kernely_onfly_R.rand_vypln(0.0,0.01);
    biaskonv_onfly_Q.clear();
    biaskonv_onfly_R.clear();

    Tenzor<double> akt_vstup_Q(1,row_ker,col_ker);
    Tenzor<double> akt_vstup_R(1,row_ker,col_ker);
    Tenzor<double> deltazmlp(1,1,1);
    Tenzor<double> uprava_k_Q(poc_ker,row_ker,col_ker);
    Tenzor<double> uprava_k_R(poc_ker,row_ker,col_ker);
    Tenzor<double> vrstva_vystup_Q;
    Tenzor<double> vrstva_vystup_R;
    std::vector<double> vystzkonv;
     
    for(int i = 0; i<poc_ker;i++){
        biaskonv_onfly_Q.push_back(0.0);
    }

    for(int i = 0; i<poc_ker;i++){
        biaskonv_onfly_R.push_back(0.0);
    }


//////////////////////////////////////////////////////////// KALIBRACE //////////////////////////////////////////////////////
    for(int m = 0; m < iter; m++){
//        std::cout<<m<<"\n";
        for(int roky = 0; roky < (dataprocnn_Q.getRows() - (row_ker - 1)); roky++){
            for(int dny = 0; dny < rok ; dny++){

//// KONVOLUCE //////////////////////////////////////////////////////////////////////////////////////////////////////////////                
                for(int i = 0; i < row_ker; i++){
                    for(int j = 0; j < col_ker; j++){
                        akt_vstup_Q.setElement(0,i,j,dataprocnn_Q.getElement(i+roky,j+dny));
                    }
                }

                for(int i = 0; i < row_ker; i++){
                    for(int j = 0; j < col_ker; j++){
                        akt_vstup_R.setElement(0,i,j,dataprocnn_R.getElement(i+roky,j+dny));
                    }
                }

                vrstva_vystup_Q = konvo_3d(akt_vstup_Q,kernely_onfly_Q);
                vrstva_vystup_R = konvo_3d(akt_vstup_R,kernely_onfly_R);

                for (int i = 0; i<vrstva_vystup_Q.getDepth();i++){
                    vrstva_vystup_Q.setElement(i,0,0,(vrstva_vystup_Q.getElement(i,0,0) + biaskonv_onfly_Q[i]));
                }

                for (int i = 0; i<vrstva_vystup_R.getDepth();i++){
                    vrstva_vystup_R.setElement(i,0,0,(vrstva_vystup_R.getElement(i,0,0) + biaskonv_onfly_R[i]));
                }
                
                vystzkonv.clear();
                for(int i = 0;i<vrstva_vystup_Q.getDepth();++i){
                    if(vrstva_vystup_Q.getElement(i,0,0) < 0.0){
                        vystzkonv.push_back(0.01 * vrstva_vystup_Q.getElement(i,0,0));
                        } else {
                            vystzkonv.push_back(vrstva_vystup_Q.getElement(i,0,0));
                        }
                        
                }
                for(int i = 0;i<vrstva_vystup_R.getDepth();++i){
                    if(vrstva_vystup_R.getElement(i,0,0) < 0.0){
                        vystzkonv.push_back(0.01 * vrstva_vystup_R.getElement(i,0,0));
                        } else {
                            vystzkonv.push_back(vrstva_vystup_R.getElement(i,0,0));
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

//// CNN BACKPROP ////////////////////////////////////////////////////////////////////////////////////////////////////////////
                for (int neur = 0;neur<rozmery[0];++neur){
                    deltazmlp.setElement(0,0,0,sit[0][neur].delta);
                    uprava_k_Q = konvo_3d(akt_vstup_Q,deltazmlp); 
                    uprava_k_R = konvo_3d(akt_vstup_R,deltazmlp); 
                    for(int upr = 0;upr<kernely_onfly_Q.getDepth();++upr){
                        for(int sl = 0; sl<kernely_onfly_Q.getCols();++sl){
                            for(int rad = 0;rad<kernely_onfly_Q.getRows();++rad){
                                
                                if(vystzkonv[upr] < 0.0){
                                    kernely_onfly_Q.setElement(upr,rad,sl,kernely_onfly_Q.getElement(upr,rad,sl)-alfa*0.01*uprava_k_Q.getElement(0,rad,sl));
                                } else{
                                    kernely_onfly_Q.setElement(upr,rad,sl,kernely_onfly_Q.getElement(upr,rad,sl)-alfa*uprava_k_Q.getElement(0,rad,sl));
                                }
                            }
                        }
                        if(vystzkonv[upr] < 0.0){
                            biaskonv_onfly_Q[upr] = biaskonv_onfly_Q[upr] - alfa*0.01*sit[0][neur].delta;
                        } else {
                            biaskonv_onfly_Q[upr] = biaskonv_onfly_Q[upr] - alfa*sit[0][neur].delta;
                        }
                    }

                    for(int upr = 0;upr<kernely_onfly_R.getDepth();++upr){
                        for(int sl = 0; sl<kernely_onfly_R.getCols();++sl){
                            for(int rad = 0;rad<kernely_onfly_R.getRows();++rad){
                                
                                if(vystzkonv[upr+poc_ker] < 0.0){
                                    kernely_onfly_R.setElement(upr,rad,sl,kernely_onfly_R.getElement(upr,rad,sl)-alfa*0.01*uprava_k_R.getElement(0,rad,sl));
                                } else{
                                    kernely_onfly_R.setElement(upr,rad,sl,kernely_onfly_R.getElement(upr,rad,sl)-alfa*uprava_k_R.getElement(0,rad,sl));
                                }
                            }
                        }
                        if(vystzkonv[upr+poc_ker] < 0.0){
                            biaskonv_onfly_R[upr] = biaskonv_onfly_R[upr] - alfa*0.01*sit[0][neur].delta;
                        } else {
                            biaskonv_onfly_R[upr] = biaskonv_onfly_R[upr] - alfa*sit[0][neur].delta;
                        }
                    }
                }
            }
        }
    }
} else if (velic == 3){
    Matice<double> dataprocnn_Q;
    Matice<double> dataprocnn_R;
    Matice<double> dataprocnn_T;
    dataprocnn_Q = udelej_radky(rok,Q_kal_vstup);
    dataprocnn_Q.sloupce_nakonec(col_ker - 1);
    dataprocnn_R = udelej_radky(rok,R_kal_vstup);
    dataprocnn_R.sloupce_nakonec(col_ker - 1);
    dataprocnn_T = udelej_radky(rok,T_kal_vstup);
    dataprocnn_T.sloupce_nakonec(col_ker - 1);
    kernely_onfly_Q.resize(poc_ker,row_ker,col_ker);
    kernely_onfly_Q.rand_vypln(0.0,0.1);
    kernely_onfly_R.resize(poc_ker,row_ker,col_ker);
    kernely_onfly_R.rand_vypln(0.0,0.01);
    kernely_onfly_T.resize(poc_ker,row_ker,col_ker);
    kernely_onfly_T.rand_vypln(0.0,0.001);
    biaskonv_onfly_Q.clear();
    biaskonv_onfly_R.clear();
    biaskonv_onfly_T.clear();

    Tenzor<double> akt_vstup_Q(1,row_ker,col_ker);
    Tenzor<double> akt_vstup_R(1,row_ker,col_ker);
    Tenzor<double> akt_vstup_T(1,row_ker,col_ker);
    Tenzor<double> deltazmlp(1,1,1);
    Tenzor<double> uprava_k_Q(poc_ker,row_ker,col_ker);
    Tenzor<double> uprava_k_R(poc_ker,row_ker,col_ker);
    Tenzor<double> uprava_k_T(poc_ker,row_ker,col_ker);
    Tenzor<double> vrstva_vystup_Q;
    Tenzor<double> vrstva_vystup_R;
    Tenzor<double> vrstva_vystup_T;
    std::vector<double> vystzkonv;
     
    for(int i = 0; i<poc_ker;i++){
        biaskonv_onfly_Q.push_back(0.0);
        biaskonv_onfly_R.push_back(0.0);
        biaskonv_onfly_T.push_back(0.0);
    }

//////////////////////////////////////////////////////////// KALIBRACE //////////////////////////////////////////////////////
    for(int m = 0; m < iter; m++){
//        std::cout<<m<<"\n";
        for(int roky = 0; roky < (dataprocnn_Q.getRows() - (row_ker - 1)); roky++){
            for(int dny = 0; dny < rok ; dny++){

//// KONVOLUCE //////////////////////////////////////////////////////////////////////////////////////////////////////////////                
                for(int i = 0; i < row_ker; i++){
                    for(int j = 0; j < col_ker; j++){
                        akt_vstup_Q.setElement(0,i,j,dataprocnn_Q.getElement(i+roky,j+dny));
                        akt_vstup_R.setElement(0,i,j,dataprocnn_R.getElement(i+roky,j+dny));
                        akt_vstup_T.setElement(0,i,j,dataprocnn_T.getElement(i+roky,j+dny));
                    }
                }

                vrstva_vystup_Q = konvo_3d(akt_vstup_Q,kernely_onfly_Q);
                vrstva_vystup_R = konvo_3d(akt_vstup_R,kernely_onfly_R);
                vrstva_vystup_T = konvo_3d(akt_vstup_T,kernely_onfly_T);

                for (int i = 0; i<vrstva_vystup_Q.getDepth();i++){
                    vrstva_vystup_Q.setElement(i,0,0,(vrstva_vystup_Q.getElement(i,0,0) + biaskonv_onfly_Q[i]));
                }
                for (int i = 0; i<vrstva_vystup_R.getDepth();i++){
                    vrstva_vystup_R.setElement(i,0,0,(vrstva_vystup_R.getElement(i,0,0) + biaskonv_onfly_R[i]));
                }
                for (int i = 0; i<vrstva_vystup_T.getDepth();i++){
                    vrstva_vystup_T.setElement(i,0,0,(vrstva_vystup_T.getElement(i,0,0) + biaskonv_onfly_T[i]));
                }
                
                vystzkonv.clear();
                for(int i = 0;i<vrstva_vystup_Q.getDepth();++i){
                    if(vrstva_vystup_Q.getElement(i,0,0) < 0.0){
                        vystzkonv.push_back(0.01 * vrstva_vystup_Q.getElement(i,0,0));
                        } else {
                            vystzkonv.push_back(vrstva_vystup_Q.getElement(i,0,0));
                        }
                        
                }
                for(int i = 0;i<vrstva_vystup_R.getDepth();++i){
                    if(vrstva_vystup_R.getElement(i,0,0) < 0.0){
                        vystzkonv.push_back(0.01 * vrstva_vystup_R.getElement(i,0,0));
                        } else {
                            vystzkonv.push_back(vrstva_vystup_R.getElement(i,0,0));
                        }
                        
                }
                for(int i = 0;i<vrstva_vystup_T.getDepth();++i){
                    if(vrstva_vystup_T.getElement(i,0,0) < 0.0){
                        vystzkonv.push_back(0.01 * vrstva_vystup_T.getElement(i,0,0));
                        } else {
                            vystzkonv.push_back(vrstva_vystup_T.getElement(i,0,0));
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

//// CNN BACKPROP ////////////////////////////////////////////////////////////////////////////////////////////////////////////
                for (int neur = 0;neur<rozmery[0];++neur){
                    deltazmlp.setElement(0,0,0,sit[0][neur].delta);
                    uprava_k_Q = konvo_3d(akt_vstup_Q,deltazmlp); 
                    uprava_k_R = konvo_3d(akt_vstup_R,deltazmlp);
                    uprava_k_T = konvo_3d(akt_vstup_T,deltazmlp); 
                    for(int upr = 0;upr<kernely_onfly_Q.getDepth();++upr){
                        for(int sl = 0; sl<kernely_onfly_Q.getCols();++sl){
                            for(int rad = 0;rad<kernely_onfly_Q.getRows();++rad){
                                
                                if(vystzkonv[upr] < 0.0){
                                    kernely_onfly_Q.setElement(upr,rad,sl,kernely_onfly_Q.getElement(upr,rad,sl)-alfa*0.01*uprava_k_Q.getElement(0,rad,sl));
                                } else{
                                    kernely_onfly_Q.setElement(upr,rad,sl,kernely_onfly_Q.getElement(upr,rad,sl)-alfa*uprava_k_Q.getElement(0,rad,sl));
                                }
                            }
                        }
                        if(vystzkonv[upr] < 0.0){
                            biaskonv_onfly_Q[upr] = biaskonv_onfly_Q[upr] - alfa*0.01*sit[0][neur].delta;
                        } else {
                            biaskonv_onfly_Q[upr] = biaskonv_onfly_Q[upr] - alfa*sit[0][neur].delta;
                        }
                    }

                    for(int upr = 0;upr<kernely_onfly_R.getDepth();++upr){
                        for(int sl = 0; sl<kernely_onfly_R.getCols();++sl){
                            for(int rad = 0;rad<kernely_onfly_R.getRows();++rad){
                                
                                if(vystzkonv[upr+poc_ker] < 0.0){
                                    kernely_onfly_R.setElement(upr,rad,sl,kernely_onfly_R.getElement(upr,rad,sl)-alfa*0.01*uprava_k_R.getElement(0,rad,sl));
                                } else{
                                    kernely_onfly_R.setElement(upr,rad,sl,kernely_onfly_R.getElement(upr,rad,sl)-alfa*uprava_k_R.getElement(0,rad,sl));
                                }
                            }
                        }
                        if(vystzkonv[upr+poc_ker] < 0.0){
                            biaskonv_onfly_R[upr] = biaskonv_onfly_R[upr] - alfa*0.01*sit[0][neur].delta;
                        } else {
                            biaskonv_onfly_R[upr] = biaskonv_onfly_R[upr] - alfa*sit[0][neur].delta;
                        }
                    }

                    for(int upr = 0;upr<kernely_onfly_T.getDepth();++upr){
                        for(int sl = 0; sl<kernely_onfly_T.getCols();++sl){
                            for(int rad = 0;rad<kernely_onfly_T.getRows();++rad){
                                
                                if(vystzkonv[upr+2*poc_ker] < 0.0){
                                    kernely_onfly_T.setElement(upr,rad,sl,kernely_onfly_T.getElement(upr,rad,sl)-alfa*0.01*uprava_k_T.getElement(0,rad,sl));
                                } else{
                                    kernely_onfly_T.setElement(upr,rad,sl,kernely_onfly_T.getElement(upr,rad,sl)-alfa*uprava_k_T.getElement(0,rad,sl));
                                }
                            }
                        }
                        if(vystzkonv[upr+2*poc_ker] < 0.0){
                            biaskonv_onfly_T[upr] = biaskonv_onfly_T[upr] - alfa*0.01*sit[0][neur].delta;
                        } else {
                            biaskonv_onfly_T[upr] = biaskonv_onfly_T[upr] - alfa*sit[0][neur].delta;
                        }
                    }
                }
            }
        }
    }
}
}

void NN::cnnonfly_val(int velic){
    if(velic == 2){
    Matice<double> dataprocnn_Q;
    Matice<double> dataprocnn_R;
    int rok = 365;
    int row_ker = kernely_onfly_Q.getRows();
    int col_ker = kernely_onfly_Q.getCols();
    dataprocnn_Q = udelej_radky(rok,Q_val_vstup);
    dataprocnn_Q.sloupce_nakonec(col_ker - 1);
    dataprocnn_R = udelej_radky(rok,R_val_vstup);
    dataprocnn_R.sloupce_nakonec(col_ker - 1);
    Tenzor<double> akt_vstup_Q(1,row_ker,col_ker);
    Tenzor<double> vrstva_vystup_Q;
    Tenzor<double> akt_vstup_R(1,row_ker,col_ker);
    Tenzor<double> vrstva_vystup_R;
    std::vector<double> vystzkonv;

    vystupy.clear();
    for(int roky = 0; roky < (dataprocnn_Q.getRows() - (row_ker - 1)); roky++){
            for(int dny = 0; dny < rok ; dny++){

//// KONVOLUCE //////////////////////////////////////////////////////////////////////////////////////////////////////////////                
                for(int i = 0; i < row_ker; i++){
                    for(int j = 0; j < col_ker; j++){
                        akt_vstup_Q.setElement(0,i,j,dataprocnn_Q.getElement(i+roky,j+dny));
                        akt_vstup_R.setElement(0,i,j,dataprocnn_R.getElement(i+roky,j+dny));
                    }
                }

                vrstva_vystup_Q = konvo_3d(akt_vstup_Q,kernely_onfly_Q);
                vrstva_vystup_R = konvo_3d(akt_vstup_R,kernely_onfly_R);
                for (int i = 0; i<vrstva_vystup_Q.getDepth();i++){
                    vrstva_vystup_Q.setElement(i,0,0,(vrstva_vystup_Q.getElement(i,0,0) + biaskonv_onfly_Q[i]));
                }
                for (int i = 0; i<vrstva_vystup_R.getDepth();i++){
                    vrstva_vystup_R.setElement(i,0,0,(vrstva_vystup_R.getElement(i,0,0) + biaskonv_onfly_R[i]));
                }
                vystzkonv.clear();
                for(int i = 0;i<vrstva_vystup_Q.getDepth();++i){
                    if(vrstva_vystup_Q.getElement(i,0,0) < 0.0){
                        vystzkonv.push_back(0.01 * vrstva_vystup_Q.getElement(i,0,0));
                        } else {
                            vystzkonv.push_back(vrstva_vystup_Q.getElement(i,0,0));
                        }
                        
                }
                for(int i = 0;i<vrstva_vystup_R.getDepth();++i){
                    if(vrstva_vystup_R.getElement(i,0,0) < 0.0){
                        vystzkonv.push_back(0.01 * vrstva_vystup_R.getElement(i,0,0));
                        } else {
                            vystzkonv.push_back(vrstva_vystup_R.getElement(i,0,0));
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
    }else if(velic == 3){
        Matice<double> dataprocnn_Q;
        Matice<double> dataprocnn_R;
        Matice<double> dataprocnn_T;
        int rok = 365;
        int row_ker = kernely_onfly_Q.getRows();
        int col_ker = kernely_onfly_Q.getCols();
        dataprocnn_Q = udelej_radky(rok,Q_val_vstup);
        dataprocnn_Q.sloupce_nakonec(col_ker - 1);
        dataprocnn_R = udelej_radky(rok,R_val_vstup);
        dataprocnn_R.sloupce_nakonec(col_ker - 1);
        dataprocnn_T = udelej_radky(rok,T_val_vstup);
        dataprocnn_T.sloupce_nakonec(col_ker - 1);
        Tenzor<double> akt_vstup_Q(1,row_ker,col_ker);
        Tenzor<double> vrstva_vystup_Q;
        Tenzor<double> akt_vstup_R(1,row_ker,col_ker);
        Tenzor<double> vrstva_vystup_R;
        Tenzor<double> akt_vstup_T(1,row_ker,col_ker);
        Tenzor<double> vrstva_vystup_T;
        std::vector<double> vystzkonv;
    
        vystupy.clear();
        for(int roky = 0; roky < (dataprocnn_Q.getRows() - (row_ker - 1)); roky++){
                for(int dny = 0; dny < rok ; dny++){
    
    //// KONVOLUCE //////////////////////////////////////////////////////////////////////////////////////////////////////////////                
                    for(int i = 0; i < row_ker; i++){
                        for(int j = 0; j < col_ker; j++){
                            akt_vstup_Q.setElement(0,i,j,dataprocnn_Q.getElement(i+roky,j+dny));
                            akt_vstup_R.setElement(0,i,j,dataprocnn_R.getElement(i+roky,j+dny));
                            akt_vstup_T.setElement(0,i,j,dataprocnn_T.getElement(i+roky,j+dny));
                        }
                    }
    
                    vrstva_vystup_Q = konvo_3d(akt_vstup_Q,kernely_onfly_Q);
                    vrstva_vystup_R = konvo_3d(akt_vstup_R,kernely_onfly_R);
                    vrstva_vystup_T = konvo_3d(akt_vstup_T,kernely_onfly_T);
                    for (int i = 0; i<vrstva_vystup_Q.getDepth();i++){
                        vrstva_vystup_Q.setElement(i,0,0,(vrstva_vystup_Q.getElement(i,0,0) + biaskonv_onfly_Q[i]));
                    }
                    for (int i = 0; i<vrstva_vystup_R.getDepth();i++){
                        vrstva_vystup_R.setElement(i,0,0,(vrstva_vystup_R.getElement(i,0,0) + biaskonv_onfly_R[i]));
                    }
                    for (int i = 0; i<vrstva_vystup_T.getDepth();i++){
                        vrstva_vystup_T.setElement(i,0,0,(vrstva_vystup_T.getElement(i,0,0) + biaskonv_onfly_T[i]));
                    }
                    vystzkonv.clear();
                    for(int i = 0;i<vrstva_vystup_Q.getDepth();++i){
                        if(vrstva_vystup_Q.getElement(i,0,0) < 0.0){
                            vystzkonv.push_back(0.01 * vrstva_vystup_Q.getElement(i,0,0));
                            } else {
                                vystzkonv.push_back(vrstva_vystup_Q.getElement(i,0,0));
                            }
                            
                    }
                    for(int i = 0;i<vrstva_vystup_R.getDepth();++i){
                        if(vrstva_vystup_R.getElement(i,0,0) < 0.0){
                            vystzkonv.push_back(0.01 * vrstva_vystup_R.getElement(i,0,0));
                            } else {
                                vystzkonv.push_back(vrstva_vystup_R.getElement(i,0,0));
                            }
                            
                    }
                    for(int i = 0;i<vrstva_vystup_T.getDepth();++i){
                        if(vrstva_vystup_T.getElement(i,0,0) < 0.0){
                            vystzkonv.push_back(0.01 * vrstva_vystup_T.getElement(i,0,0));
                            } else {
                                vystzkonv.push_back(vrstva_vystup_T.getElement(i,0,0));
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
}

void NN::cnn_full_cal(int iter, int velic){
alfa = 0.0001;
    if(velic == 2){
    kernely_full_1_Q.resize(10,3,3);
    kernely_full_1_Q.rand_vypln(0.0,0.1);
    kernely_full_2_Q.resize(5,2,2);
    kernely_full_2_Q.rand_vypln(0.0,0.1);
    for (int k1_depth = 0;k1_depth<kernely_full_1_Q.getDepth();k1_depth++){
        bias_full_k1_Q.push_back(0);
    }
    for (int k2_depth = 0;k2_depth<kernely_full_2_Q.getDepth();k2_depth++){
        bias_full_k2_Q.push_back(0);
    }

    kernely_full_1_R.resize(10,3,3);
    kernely_full_1_R.rand_vypln(0.0,0.01);
    kernely_full_2_R.resize(5,2,2);
    kernely_full_2_R.rand_vypln(0.0,0.01);
    for (int k1_depth = 0;k1_depth<kernely_full_1_R.getDepth();k1_depth++){
        bias_full_k1_R.push_back(0);
    }
    for (int k2_depth = 0;k2_depth<kernely_full_2_R.getDepth();k2_depth++){
        bias_full_k2_R.push_back(0);
    }

    ////////////////////////////////KONVOLUCE
    Tenzor<double> vrstva0_Q;
    Tenzor<double> vrstva1_Q;
    Tenzor<double> vrstva2_Q;
    Tenzor<double> vrstva_final_Q;
    Tenzor<double> grad_Q;
    Tenzor<double> uprava_k2_Q;
    Matice<double> dataprocnn_Q;
    std::vector<double> current_kus_Q;

    Tenzor<double> vrstva0_R;
    Tenzor<double> vrstva1_R;
    Tenzor<double> vrstva2_R;
    Tenzor<double> vrstva_final_R;
    Tenzor<double> grad_R;
    Tenzor<double> uprava_k2_R;
    Matice<double> dataprocnn_R;
    std::vector<double> current_kus_R;

    std::vector<double> vystzkonv;
    Tenzor<double> deltazmlp(1,1,1);

for(int ite = 0;ite<iter;++ite){
    std::cout<<ite;
    for (int kroky = 0; kroky < (Q_kal_vstup.size() - 36); kroky++){
        vrstva0_Q.resize(0,0,0);
        vrstva1_Q.resize(0,0,0);
        vrstva2_Q.resize(0,0,0);
        vrstva_final_Q.resize(0,0,0);

        for(int kus = 0; kus < 36; kus++){
            current_kus_Q.push_back(Q_kal_vstup[kroky + kus]);
        }
        chtenejout.clear();
        chtenejout.push_back(Q_kal_vstup[36+kroky]);


        dataprocnn_Q = udelej_radky(6,current_kus_Q);
        current_kus_Q.clear();
        vrstva0_Q.add_matrix(dataprocnn_Q);
        dataprocnn_Q.resize(0,0);
        vrstva1_Q = konvo_3d(vrstva0_Q,kernely_full_1_Q);
        for(int dep_v1 = 0;dep_v1<vrstva1_Q.getDepth();dep_v1++){
            for(int row_v1 = 0;row_v1<vrstva1_Q.getRows();row_v1++){
                for(int col_v1 = 0;col_v1<vrstva1_Q.getCols();col_v1++){
                    vrstva1_Q.setElement(dep_v1,row_v1,col_v1,(vrstva1_Q.getElement(dep_v1,row_v1,col_v1)+bias_full_k1_Q[dep_v1]));
                    if(vrstva1_Q.getElement(dep_v1,row_v1,col_v1)<0.0){
                        vrstva1_Q.setElement(dep_v1,row_v1,col_v1,(vrstva1_Q.getElement(dep_v1,row_v1,col_v1)*0.01));
                    }
                }
            }
        }
        vrstva2_Q = max_pool_fullstep_3d(vrstva1_Q,2,2);
        vrstva_final_Q = konvo_3d(vrstva2_Q,kernely_full_2_Q);
        for(int dep_v2=0;dep_v2<vrstva2_Q.getDepth();dep_v2++){
            for(int dep_k2=0;dep_k2<kernely_full_2_Q.getDepth();dep_k2++){
                vrstva_final_Q.setElement((dep_v2*kernely_full_2_Q.getDepth()+dep_k2),0,0,(vrstva_final_Q.getElement((dep_v2*kernely_full_2_Q.getDepth()+dep_k2),0,0)+bias_full_k2_Q[dep_k2]));
                if(vrstva_final_Q.getElement((dep_v2*kernely_full_2_Q.getDepth()+dep_k2),0,0)<0.0){
                    vrstva_final_Q.setElement((dep_v2*kernely_full_2_Q.getDepth()+dep_k2),0,0,(vrstva_final_Q.getElement((dep_v2*kernely_full_2_Q.getDepth()+dep_k2),0,0)*0.01));
                }
            }
        }


        vrstva0_R.resize(0,0,0);
        vrstva1_R.resize(0,0,0);
        vrstva2_R.resize(0,0,0);
        vrstva_final_R.resize(0,0,0);

        for(int kus = 0; kus < 36; kus++){
            current_kus_R.push_back(R_kal_vstup[kroky + kus]);
        }

        dataprocnn_R = udelej_radky(6,current_kus_R);
        current_kus_R.clear();
        vrstva0_R.add_matrix(dataprocnn_R);
        dataprocnn_R.resize(0,0);
        vrstva1_R = konvo_3d(vrstva0_R,kernely_full_1_R);
        for(int dep_v1 = 0;dep_v1<vrstva1_R.getDepth();dep_v1++){
            for(int row_v1 = 0;row_v1<vrstva1_R.getRows();row_v1++){
                for(int col_v1 = 0;col_v1<vrstva1_R.getCols();col_v1++){
                    vrstva1_R.setElement(dep_v1,row_v1,col_v1,(vrstva1_R.getElement(dep_v1,row_v1,col_v1)+bias_full_k1_R[dep_v1]));
                    if(vrstva1_R.getElement(dep_v1,row_v1,col_v1)<0.0){
                        vrstva1_R.setElement(dep_v1,row_v1,col_v1,(vrstva1_R.getElement(dep_v1,row_v1,col_v1)*0.01));
                    }
                }
            }
        }
        vrstva2_R = max_pool_fullstep_3d(vrstva1_R,2,2);
        vrstva_final_R = konvo_3d(vrstva2_R,kernely_full_2_R);
        for(int dep_v2=0;dep_v2<vrstva2_R.getDepth();dep_v2++){
            for(int dep_k2=0;dep_k2<kernely_full_2_R.getDepth();dep_k2++){
                vrstva_final_R.setElement((dep_v2*kernely_full_2_R.getDepth()+dep_k2),0,0,(vrstva_final_R.getElement((dep_v2*kernely_full_2_R.getDepth()+dep_k2),0,0)+bias_full_k2_R[dep_k2]));
                if(vrstva_final_R.getElement((dep_v2*kernely_full_2_R.getDepth()+dep_k2),0,0)<0.0){
                    vrstva_final_R.setElement((dep_v2*kernely_full_2_R.getDepth()+dep_k2),0,0,(vrstva_final_R.getElement((dep_v2*kernely_full_2_R.getDepth()+dep_k2),0,0)*0.01));
                }
            }
        }

    ///////////////////////////MLP

    vystupy.clear();
    vystzkonv.clear();

    for(int i = 0;i<vrstva_final_Q.getDepth();++i){
        vystzkonv.push_back(vrstva_final_Q.getElement(i,0,0));
    }

    for(int i = 0;i<vrstva_final_R.getDepth();++i){
        vystzkonv.push_back(vrstva_final_R.getElement(i,0,0));
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
        for(int i = 0;i<rozmery[pocet_vrstev-1];i++){
            sit[pocet_vrstev-1][i].delta = vystupy[i] - chtenejout[i];
        }

        for(int j = (pocet_vrstev-2); j>=0;--j){
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

    grad_Q.resize(0,0,0);
    grad_Q = kernely_full_2_Q;
    grad_Q.flip180();
    grad_Q = konvo_3d(grad_Q,deltazmlp);

    grad_R.resize(0,0,0);
    grad_R = kernely_full_2_R;
    grad_R.flip180();
    grad_R = konvo_3d(grad_R,deltazmlp);

    uprava_k2_Q = konvo_3d(vrstva2_Q,deltazmlp);
    uprava_k2_R = konvo_3d(vrstva2_R,deltazmlp);
    
    for(int upr = 0;upr<uprava_k2_Q.getDepth();++upr){
        for(int ker = 0;ker<kernely_full_2_Q.getDepth();++ker){
            for(int sl = 0; sl<kernely_full_2_Q.getCols();++sl){
                for(int rad = 0;rad<kernely_full_2_Q.getRows();++rad){
                    if(vrstva_final_Q.getElement((upr*kernely_full_2_Q.getDepth()+ker),0,0)<0.0){
                        kernely_full_2_Q.setElement(ker,rad,sl,kernely_full_2_Q.getElement(ker,rad,sl)-alfa*0.01* uprava_k2_Q.getElement(upr,rad,sl));
                        bias_full_k2_Q[ker] = bias_full_k2_Q[ker] - alfa * 0.01 * deltazmlp.getElement(0,0,0);
                    }else{
                        kernely_full_2_Q.setElement(ker,rad,sl,kernely_full_2_Q.getElement(ker,rad,sl)-alfa*uprava_k2_Q.getElement(upr,rad,sl));
                        bias_full_k2_Q[ker] = bias_full_k2_Q[ker] - alfa* deltazmlp.getElement(0,0,0);
                    }
                }
            }
        }
    }
    uprava_k2_Q.resize(0,0,0);

    for(int upr = 0;upr<uprava_k2_R.getDepth();++upr){
        for(int ker = 0;ker<kernely_full_2_R.getDepth();++ker){
            for(int sl = 0; sl<kernely_full_2_R.getCols();++sl){
                for(int rad = 0;rad<kernely_full_2_R.getRows();++rad){
                    if(vrstva_final_R.getElement((upr*kernely_full_2_R.getDepth()+ker),0,0)<0.0){
                        kernely_full_2_R.setElement(ker,rad,sl,kernely_full_2_R.getElement(ker,rad,sl)-alfa*0.01* uprava_k2_R.getElement(upr,rad,sl));
                        bias_full_k2_R[ker] = bias_full_k2_R[ker] - alfa * 0.01 * deltazmlp.getElement(0,0,0);
                    }else{
                        kernely_full_2_R.setElement(ker,rad,sl,kernely_full_2_R.getElement(ker,rad,sl)-alfa*uprava_k2_R.getElement(upr,rad,sl));
                        bias_full_k2_R[ker] = bias_full_k2_R[ker] - alfa* deltazmlp.getElement(0,0,0);
                    }
                }
            }
        }
    }
    uprava_k2_R.resize(0,0,0);

    for(int grad_depth = 0; grad_depth<grad_Q.getDepth();grad_depth++){
        for(int uk1 = 0; uk1<kernely_full_1_Q.getDepth();uk1++){    
            for(int rad_v2 = 0; rad_v2<vrstva2_Q.getRows(); rad_v2++){
                for(int sl_v2 = 0; sl_v2<vrstva2_Q.getCols();sl_v2++){
                    for(int rad_v1 = 0; rad_v1<vrstva2_Q.getRows(); rad_v1++){
                        for(int sl_v1 = 0; sl_v1<vrstva2_Q.getCols();sl_v1++){
                            if(vrstva2_Q.getElement(uk1,rad_v2,sl_v2) == vrstva1_Q.getElement(uk1,(rad_v2*vrstva2_Q.getRows()+rad_v1),(sl_v2*vrstva2_Q.getCols()+sl_v1))){
                                for(int rad_ker = 0; rad_ker<kernely_full_1_Q.getRows();rad_ker++){
                                    for(int sl_ker = 0; sl_ker<kernely_full_1_Q.getCols();sl_ker++){
                                        if(vrstva2_Q.getElement(uk1,rad_v2,sl_v2)<0.0){
                                            kernely_full_1_Q.setElement(uk1,rad_ker,sl_ker,(kernely_full_1_Q.getElement(uk1,rad_ker,sl_ker) - alfa * 0.01*grad_Q.getElement(grad_depth,rad_v2,sl_v2)*vrstva0_Q.getElement(0,(rad_v2*vrstva2_Q.getRows()+rad_v1+rad_ker),(sl_v2*vrstva2_Q.getCols()+sl_v1+sl_ker))));
                                            bias_full_k1_Q[uk1] = bias_full_k1_Q[uk1] - alfa * 0.01*grad_Q.getElement(grad_depth,rad_v2,sl_v2);
                                        }else{
                                            kernely_full_1_Q.setElement(uk1,rad_ker,sl_ker,(kernely_full_1_Q.getElement(uk1,rad_ker,sl_ker) - alfa*grad_Q.getElement(grad_depth,rad_v2,sl_v2)*vrstva0_Q.getElement(0,(rad_v2*vrstva2_Q.getRows()+rad_v1+rad_ker),(sl_v2*vrstva2_Q.getCols()+sl_v1+sl_ker))));
                                            bias_full_k1_Q[uk1] = bias_full_k1_Q[uk1] - alfa*grad_Q.getElement(grad_depth,rad_v2,sl_v2);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    for(int grad_depth = 0; grad_depth<grad_R.getDepth();grad_depth++){
        for(int uk1 = 0; uk1<kernely_full_1_R.getDepth();uk1++){    
            for(int rad_v2 = 0; rad_v2<vrstva2_R.getRows(); rad_v2++){
                for(int sl_v2 = 0; sl_v2<vrstva2_R.getCols();sl_v2++){
                    for(int rad_v1 = 0; rad_v1<vrstva2_R.getRows(); rad_v1++){
                        for(int sl_v1 = 0; sl_v1<vrstva2_R.getCols();sl_v1++){
                            if(vrstva2_R.getElement(uk1,rad_v2,sl_v2) == vrstva1_R.getElement(uk1,(rad_v2*vrstva2_R.getRows()+rad_v1),(sl_v2*vrstva2_R.getCols()+sl_v1))){
                                for(int rad_ker = 0; rad_ker<kernely_full_1_R.getRows();rad_ker++){
                                    for(int sl_ker = 0; sl_ker<kernely_full_1_R.getCols();sl_ker++){
                                        if(vrstva2_R.getElement(uk1,rad_v2,sl_v2)<0.0){
                                            kernely_full_1_R.setElement(uk1,rad_ker,sl_ker,(kernely_full_1_R.getElement(uk1,rad_ker,sl_ker) - alfa * 0.01*grad_R.getElement(grad_depth,rad_v2,sl_v2)*vrstva0_R.getElement(0,(rad_v2*vrstva2_R.getRows()+rad_v1+rad_ker),(sl_v2*vrstva2_R.getCols()+sl_v1+sl_ker))));
                                            bias_full_k1_R[uk1] = bias_full_k1_R[uk1] - alfa * 0.01*grad_R.getElement(grad_depth,rad_v2,sl_v2);
                                        }else{
                                            kernely_full_1_R.setElement(uk1,rad_ker,sl_ker,(kernely_full_1_R.getElement(uk1,rad_ker,sl_ker) - alfa*grad_R.getElement(grad_depth,rad_v2,sl_v2)*vrstva0_R.getElement(0,(rad_v2*vrstva2_R.getRows()+rad_v1+rad_ker),(sl_v2*vrstva2_R.getCols()+sl_v1+sl_ker))));
                                            bias_full_k1_R[uk1] = bias_full_k1_R[uk1] - alfa*grad_R.getElement(grad_depth,rad_v2,sl_v2);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    }
}
}
}else if(velic == 3){
    kernely_full_1_Q.resize(10,3,3);
    kernely_full_1_Q.rand_vypln(0.0,0.1);
    kernely_full_2_Q.resize(5,2,2);
    kernely_full_2_Q.rand_vypln(0.0,0.1);
    for (int k1_depth = 0;k1_depth<kernely_full_1_Q.getDepth();k1_depth++){
        bias_full_k1_Q.push_back(0);
    }
    for (int k2_depth = 0;k2_depth<kernely_full_2_Q.getDepth();k2_depth++){
        bias_full_k2_Q.push_back(0);
    }
    kernely_full_1_R.resize(10,3,3);
    kernely_full_1_R.rand_vypln(0.0,0.01);
    kernely_full_2_R.resize(5,2,2);
    kernely_full_2_R.rand_vypln(0.0,0.01);
    for (int k1_depth = 0;k1_depth<kernely_full_1_R.getDepth();k1_depth++){
        bias_full_k1_R.push_back(0);
    }
    for (int k2_depth = 0;k2_depth<kernely_full_2_R.getDepth();k2_depth++){
        bias_full_k2_R.push_back(0);
    }
    kernely_full_1_T.resize(10,3,3);
    kernely_full_1_T.rand_vypln(0.0,0.001);
    kernely_full_2_T.resize(5,2,2);
    kernely_full_2_T.rand_vypln(0.0,0.001);
    for (int k1_depth = 0;k1_depth<kernely_full_1_T.getDepth();k1_depth++){
        bias_full_k1_T.push_back(0);
    }
    for (int k2_depth = 0;k2_depth<kernely_full_2_T.getDepth();k2_depth++){
        bias_full_k2_T.push_back(0);
    }

    ////////////////////////////////KONVOLUCE
    Tenzor<double> vrstva0_Q;
    Tenzor<double> vrstva1_Q;
    Tenzor<double> vrstva2_Q;
    Tenzor<double> vrstva_final_Q;
    Tenzor<double> grad_Q;
    Tenzor<double> uprava_k2_Q;
    Matice<double> dataprocnn_Q;
    std::vector<double> current_kus_Q;

    Tenzor<double> vrstva0_R;
    Tenzor<double> vrstva1_R;
    Tenzor<double> vrstva2_R;
    Tenzor<double> vrstva_final_R;
    Tenzor<double> grad_R;
    Tenzor<double> uprava_k2_R;
    Matice<double> dataprocnn_R;
    std::vector<double> current_kus_R;

    Tenzor<double> vrstva0_T;
    Tenzor<double> vrstva1_T;
    Tenzor<double> vrstva2_T;
    Tenzor<double> vrstva_final_T;
    Tenzor<double> grad_T;
    Tenzor<double> uprava_k2_T;
    Matice<double> dataprocnn_T;
    std::vector<double> current_kus_T;

    std::vector<double> vystzkonv;
    Tenzor<double> deltazmlp(1,1,1);

for(int ite = 0;ite<iter;++ite){
    std::cout<<ite;
    for (int kroky = 0; kroky < (Q_kal_vstup.size() - 36); kroky++){
        vrstva0_Q.resize(0,0,0);
        vrstva1_Q.resize(0,0,0);
        vrstva2_Q.resize(0,0,0);
        vrstva_final_Q.resize(0,0,0);

        for(int kus = 0; kus < 36; kus++){
            current_kus_Q.push_back(Q_kal_vstup[kroky + kus]);
        }
        chtenejout.clear();
        chtenejout.push_back(Q_kal_vstup[36+kroky]);

        dataprocnn_Q = udelej_radky(6,current_kus_Q);
        current_kus_Q.clear();
        vrstva0_Q.add_matrix(dataprocnn_Q);
        dataprocnn_Q.resize(0,0);
        vrstva1_Q = konvo_3d(vrstva0_Q,kernely_full_1_Q);
        for(int dep_v1 = 0;dep_v1<vrstva1_Q.getDepth();dep_v1++){
            for(int row_v1 = 0;row_v1<vrstva1_Q.getRows();row_v1++){
                for(int col_v1 = 0;col_v1<vrstva1_Q.getCols();col_v1++){
                    vrstva1_Q.setElement(dep_v1,row_v1,col_v1,(vrstva1_Q.getElement(dep_v1,row_v1,col_v1)+bias_full_k1_Q[dep_v1]));
                    if(vrstva1_Q.getElement(dep_v1,row_v1,col_v1)<0.0){
                        vrstva1_Q.setElement(dep_v1,row_v1,col_v1,(vrstva1_Q.getElement(dep_v1,row_v1,col_v1)*0.01));
                    }
                }
            }
        }
        vrstva2_Q = max_pool_fullstep_3d(vrstva1_Q,2,2);
        vrstva_final_Q = konvo_3d(vrstva2_Q,kernely_full_2_Q);
        for(int dep_v2=0;dep_v2<vrstva2_Q.getDepth();dep_v2++){
            for(int dep_k2=0;dep_k2<kernely_full_2_Q.getDepth();dep_k2++){
                vrstva_final_Q.setElement((dep_v2*kernely_full_2_Q.getDepth()+dep_k2),0,0,(vrstva_final_Q.getElement((dep_v2*kernely_full_2_Q.getDepth()+dep_k2),0,0)+bias_full_k2_Q[dep_k2]));
                if(vrstva_final_Q.getElement((dep_v2*kernely_full_2_Q.getDepth()+dep_k2),0,0)<0.0){
                    vrstva_final_Q.setElement((dep_v2*kernely_full_2_Q.getDepth()+dep_k2),0,0,(vrstva_final_Q.getElement((dep_v2*kernely_full_2_Q.getDepth()+dep_k2),0,0)*0.01));
                }
            }
        }


        vrstva0_R.resize(0,0,0);
        vrstva1_R.resize(0,0,0);
        vrstva2_R.resize(0,0,0);
        vrstva_final_R.resize(0,0,0);

        for(int kus = 0; kus < 36; kus++){
            current_kus_R.push_back(R_kal_vstup[kroky + kus]);
        }

        dataprocnn_R = udelej_radky(6,current_kus_R);
        current_kus_R.clear();
        vrstva0_R.add_matrix(dataprocnn_R);
        dataprocnn_R.resize(0,0);
        vrstva1_R = konvo_3d(vrstva0_R,kernely_full_1_R);
        for(int dep_v1 = 0;dep_v1<vrstva1_R.getDepth();dep_v1++){
            for(int row_v1 = 0;row_v1<vrstva1_R.getRows();row_v1++){
                for(int col_v1 = 0;col_v1<vrstva1_R.getCols();col_v1++){
                    vrstva1_R.setElement(dep_v1,row_v1,col_v1,(vrstva1_R.getElement(dep_v1,row_v1,col_v1)+bias_full_k1_R[dep_v1]));
                    if(vrstva1_R.getElement(dep_v1,row_v1,col_v1)<0.0){
                        vrstva1_R.setElement(dep_v1,row_v1,col_v1,(vrstva1_R.getElement(dep_v1,row_v1,col_v1)*0.01));
                    }
                }
            }
        }
        vrstva2_R = max_pool_fullstep_3d(vrstva1_R,2,2);
        vrstva_final_R = konvo_3d(vrstva2_R,kernely_full_2_R);
        for(int dep_v2=0;dep_v2<vrstva2_R.getDepth();dep_v2++){
            for(int dep_k2=0;dep_k2<kernely_full_2_R.getDepth();dep_k2++){
                vrstva_final_R.setElement((dep_v2*kernely_full_2_R.getDepth()+dep_k2),0,0,(vrstva_final_R.getElement((dep_v2*kernely_full_2_R.getDepth()+dep_k2),0,0)+bias_full_k2_R[dep_k2]));
                if(vrstva_final_R.getElement((dep_v2*kernely_full_2_R.getDepth()+dep_k2),0,0)<0.0){
                    vrstva_final_R.setElement((dep_v2*kernely_full_2_R.getDepth()+dep_k2),0,0,(vrstva_final_R.getElement((dep_v2*kernely_full_2_R.getDepth()+dep_k2),0,0)*0.01));
                }
            }
        }

        vrstva0_T.resize(0,0,0);
        vrstva1_T.resize(0,0,0);
        vrstva2_T.resize(0,0,0);
        vrstva_final_T.resize(0,0,0);

        for(int kus = 0; kus < 36; kus++){
            current_kus_T.push_back(T_kal_vstup[kroky + kus]);
        }

        dataprocnn_T = udelej_radky(6,current_kus_T);
        current_kus_T.clear();
        vrstva0_T.add_matrix(dataprocnn_T);
        dataprocnn_T.resize(0,0);
        vrstva1_T = konvo_3d(vrstva0_T,kernely_full_1_T);
        for(int dep_v1 = 0;dep_v1<vrstva1_T.getDepth();dep_v1++){
            for(int row_v1 = 0;row_v1<vrstva1_T.getRows();row_v1++){
                for(int col_v1 = 0;col_v1<vrstva1_T.getCols();col_v1++){
                    vrstva1_T.setElement(dep_v1,row_v1,col_v1,(vrstva1_T.getElement(dep_v1,row_v1,col_v1)+bias_full_k1_T[dep_v1]));
                    if(vrstva1_T.getElement(dep_v1,row_v1,col_v1)<0.0){
                        vrstva1_T.setElement(dep_v1,row_v1,col_v1,(vrstva1_T.getElement(dep_v1,row_v1,col_v1)*0.01));
                    }
                }
            }
        }
        vrstva2_T = max_pool_fullstep_3d(vrstva1_T,2,2);
        vrstva_final_T = konvo_3d(vrstva2_T,kernely_full_2_T);
        for(int dep_v2=0;dep_v2<vrstva2_T.getDepth();dep_v2++){
            for(int dep_k2=0;dep_k2<kernely_full_2_T.getDepth();dep_k2++){
                vrstva_final_T.setElement((dep_v2*kernely_full_2_T.getDepth()+dep_k2),0,0,(vrstva_final_T.getElement((dep_v2*kernely_full_2_T.getDepth()+dep_k2),0,0)+bias_full_k2_T[dep_k2]));
                if(vrstva_final_T.getElement((dep_v2*kernely_full_2_T.getDepth()+dep_k2),0,0)<0.0){
                    vrstva_final_T.setElement((dep_v2*kernely_full_2_T.getDepth()+dep_k2),0,0,(vrstva_final_T.getElement((dep_v2*kernely_full_2_T.getDepth()+dep_k2),0,0)*0.01));
                }
            }
        }

    ///////////////////////////MLP

    vystupy.clear();
    vystzkonv.clear();

    for(int i = 0;i<vrstva_final_Q.getDepth();++i){
        vystzkonv.push_back(vrstva_final_Q.getElement(i,0,0));
    }

    for(int i = 0;i<vrstva_final_R.getDepth();++i){
        vystzkonv.push_back(vrstva_final_R.getElement(i,0,0));
    }
    for(int i = 0;i<vrstva_final_T.getDepth();++i){
        vystzkonv.push_back(vrstva_final_T.getElement(i,0,0));
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
        for(int i = 0;i<rozmery[pocet_vrstev-1];i++){
            sit[pocet_vrstev-1][i].delta = vystupy[i] - chtenejout[i];
        }

        for(int j = (pocet_vrstev-2); j>=0;--j){
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

    grad_Q.resize(0,0,0);
    grad_Q = kernely_full_2_Q;
    grad_Q.flip180();
    grad_Q = konvo_3d(grad_Q,deltazmlp);

    grad_R.resize(0,0,0);
    grad_R = kernely_full_2_R;
    grad_R.flip180();
    grad_R = konvo_3d(grad_R,deltazmlp);

    grad_T.resize(0,0,0);
    grad_T = kernely_full_2_T;
    grad_T.flip180();
    grad_T = konvo_3d(grad_T,deltazmlp);

    uprava_k2_Q = konvo_3d(vrstva2_Q,deltazmlp);
    uprava_k2_R = konvo_3d(vrstva2_R,deltazmlp);
    uprava_k2_T = konvo_3d(vrstva2_T,deltazmlp);
    
    for(int upr = 0;upr<uprava_k2_Q.getDepth();++upr){
        for(int ker = 0;ker<kernely_full_2_Q.getDepth();++ker){
            for(int sl = 0; sl<kernely_full_2_Q.getCols();++sl){
                for(int rad = 0;rad<kernely_full_2_Q.getRows();++rad){
                    if(vrstva_final_Q.getElement((upr*kernely_full_2_Q.getDepth()+ker),0,0)<0.0){
                        kernely_full_2_Q.setElement(ker,rad,sl,kernely_full_2_Q.getElement(ker,rad,sl)-alfa*0.01* uprava_k2_Q.getElement(upr,rad,sl));
                        bias_full_k2_Q[ker] = bias_full_k2_Q[ker] - alfa * 0.01 * deltazmlp.getElement(0,0,0);
                    }else{
                        kernely_full_2_Q.setElement(ker,rad,sl,kernely_full_2_Q.getElement(ker,rad,sl)-alfa*uprava_k2_Q.getElement(upr,rad,sl));
                        bias_full_k2_Q[ker] = bias_full_k2_Q[ker] - alfa* deltazmlp.getElement(0,0,0);
                    }
                }
            }
        }
    }
    uprava_k2_Q.resize(0,0,0);

    for(int upr = 0;upr<uprava_k2_R.getDepth();++upr){
        for(int ker = 0;ker<kernely_full_2_R.getDepth();++ker){
            for(int sl = 0; sl<kernely_full_2_R.getCols();++sl){
                for(int rad = 0;rad<kernely_full_2_R.getRows();++rad){
                    if(vrstva_final_R.getElement((upr*kernely_full_2_R.getDepth()+ker),0,0)<0.0){
                        kernely_full_2_R.setElement(ker,rad,sl,kernely_full_2_R.getElement(ker,rad,sl)-alfa*0.01* uprava_k2_R.getElement(upr,rad,sl));
                        bias_full_k2_R[ker] = bias_full_k2_R[ker] - alfa * 0.01 * deltazmlp.getElement(0,0,0);
                    }else{
                        kernely_full_2_R.setElement(ker,rad,sl,kernely_full_2_R.getElement(ker,rad,sl)-alfa*uprava_k2_R.getElement(upr,rad,sl));
                        bias_full_k2_R[ker] = bias_full_k2_R[ker] - alfa* deltazmlp.getElement(0,0,0);
                    }
                }
            }
        }
    }
    uprava_k2_R.resize(0,0,0);

    for(int upr = 0;upr<uprava_k2_T.getDepth();++upr){
        for(int ker = 0;ker<kernely_full_2_T.getDepth();++ker){
            for(int sl = 0; sl<kernely_full_2_T.getCols();++sl){
                for(int rad = 0;rad<kernely_full_2_T.getRows();++rad){
                    if(vrstva_final_T.getElement((upr*kernely_full_2_T.getDepth()+ker),0,0)<0.0){
                        kernely_full_2_T.setElement(ker,rad,sl,kernely_full_2_T.getElement(ker,rad,sl)-alfa*0.01* uprava_k2_T.getElement(upr,rad,sl));
                        bias_full_k2_T[ker] = bias_full_k2_T[ker] - alfa * 0.01 * deltazmlp.getElement(0,0,0);
                    }else{
                        kernely_full_2_T.setElement(ker,rad,sl,kernely_full_2_T.getElement(ker,rad,sl)-alfa*uprava_k2_T.getElement(upr,rad,sl));
                        bias_full_k2_T[ker] = bias_full_k2_T[ker] - alfa* deltazmlp.getElement(0,0,0);
                    }
                }
            }
        }
    }
    uprava_k2_T.resize(0,0,0);

    for(int grad_depth = 0; grad_depth<grad_Q.getDepth();grad_depth++){
        for(int uk1 = 0; uk1<kernely_full_1_Q.getDepth();uk1++){    
            for(int rad_v2 = 0; rad_v2<vrstva2_Q.getRows(); rad_v2++){
                for(int sl_v2 = 0; sl_v2<vrstva2_Q.getCols();sl_v2++){
                    for(int rad_v1 = 0; rad_v1<vrstva2_Q.getRows(); rad_v1++){
                        for(int sl_v1 = 0; sl_v1<vrstva2_Q.getCols();sl_v1++){
                            if(vrstva2_Q.getElement(uk1,rad_v2,sl_v2) == vrstva1_Q.getElement(uk1,(rad_v2*vrstva2_Q.getRows()+rad_v1),(sl_v2*vrstva2_Q.getCols()+sl_v1))){
                                for(int rad_ker = 0; rad_ker<kernely_full_1_Q.getRows();rad_ker++){
                                    for(int sl_ker = 0; sl_ker<kernely_full_1_Q.getCols();sl_ker++){
                                        if(vrstva2_Q.getElement(uk1,rad_v2,sl_v2)<0.0){
                                            kernely_full_1_Q.setElement(uk1,rad_ker,sl_ker,(kernely_full_1_Q.getElement(uk1,rad_ker,sl_ker) - alfa * 0.01*grad_Q.getElement(grad_depth,rad_v2,sl_v2)*vrstva0_Q.getElement(0,(rad_v2*vrstva2_Q.getRows()+rad_v1+rad_ker),(sl_v2*vrstva2_Q.getCols()+sl_v1+sl_ker))));
                                            bias_full_k1_Q[uk1] = bias_full_k1_Q[uk1] - alfa * 0.01*grad_Q.getElement(grad_depth,rad_v2,sl_v2);
                                        }else{
                                            kernely_full_1_Q.setElement(uk1,rad_ker,sl_ker,(kernely_full_1_Q.getElement(uk1,rad_ker,sl_ker) - alfa*grad_Q.getElement(grad_depth,rad_v2,sl_v2)*vrstva0_Q.getElement(0,(rad_v2*vrstva2_Q.getRows()+rad_v1+rad_ker),(sl_v2*vrstva2_Q.getCols()+sl_v1+sl_ker))));
                                            bias_full_k1_Q[uk1] = bias_full_k1_Q[uk1] - alfa*grad_Q.getElement(grad_depth,rad_v2,sl_v2);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    for(int grad_depth = 0; grad_depth<grad_R.getDepth();grad_depth++){
        for(int uk1 = 0; uk1<kernely_full_1_R.getDepth();uk1++){    
            for(int rad_v2 = 0; rad_v2<vrstva2_R.getRows(); rad_v2++){
                for(int sl_v2 = 0; sl_v2<vrstva2_R.getCols();sl_v2++){
                    for(int rad_v1 = 0; rad_v1<vrstva2_R.getRows(); rad_v1++){
                        for(int sl_v1 = 0; sl_v1<vrstva2_R.getCols();sl_v1++){
                            if(vrstva2_R.getElement(uk1,rad_v2,sl_v2) == vrstva1_R.getElement(uk1,(rad_v2*vrstva2_R.getRows()+rad_v1),(sl_v2*vrstva2_R.getCols()+sl_v1))){
                                for(int rad_ker = 0; rad_ker<kernely_full_1_R.getRows();rad_ker++){
                                    for(int sl_ker = 0; sl_ker<kernely_full_1_R.getCols();sl_ker++){
                                        if(vrstva2_R.getElement(uk1,rad_v2,sl_v2)<0.0){
                                            kernely_full_1_R.setElement(uk1,rad_ker,sl_ker,(kernely_full_1_R.getElement(uk1,rad_ker,sl_ker) - alfa * 0.01*grad_R.getElement(grad_depth,rad_v2,sl_v2)*vrstva0_R.getElement(0,(rad_v2*vrstva2_R.getRows()+rad_v1+rad_ker),(sl_v2*vrstva2_R.getCols()+sl_v1+sl_ker))));
                                            bias_full_k1_R[uk1] = bias_full_k1_R[uk1] - alfa * 0.01*grad_R.getElement(grad_depth,rad_v2,sl_v2);
                                        }else{
                                            kernely_full_1_R.setElement(uk1,rad_ker,sl_ker,(kernely_full_1_R.getElement(uk1,rad_ker,sl_ker) - alfa*grad_R.getElement(grad_depth,rad_v2,sl_v2)*vrstva0_R.getElement(0,(rad_v2*vrstva2_R.getRows()+rad_v1+rad_ker),(sl_v2*vrstva2_R.getCols()+sl_v1+sl_ker))));
                                            bias_full_k1_R[uk1] = bias_full_k1_R[uk1] - alfa*grad_R.getElement(grad_depth,rad_v2,sl_v2);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    for(int grad_depth = 0; grad_depth<grad_T.getDepth();grad_depth++){
        for(int uk1 = 0; uk1<kernely_full_1_T.getDepth();uk1++){    
            for(int rad_v2 = 0; rad_v2<vrstva2_T.getRows(); rad_v2++){
                for(int sl_v2 = 0; sl_v2<vrstva2_T.getCols();sl_v2++){
                    for(int rad_v1 = 0; rad_v1<vrstva2_T.getRows(); rad_v1++){
                        for(int sl_v1 = 0; sl_v1<vrstva2_T.getCols();sl_v1++){
                            if(vrstva2_T.getElement(uk1,rad_v2,sl_v2) == vrstva1_T.getElement(uk1,(rad_v2*vrstva2_T.getRows()+rad_v1),(sl_v2*vrstva2_T.getCols()+sl_v1))){
                                for(int rad_ker = 0; rad_ker<kernely_full_1_T.getRows();rad_ker++){
                                    for(int sl_ker = 0; sl_ker<kernely_full_1_T.getCols();sl_ker++){
                                        if(vrstva2_T.getElement(uk1,rad_v2,sl_v2)<0.0){
                                            kernely_full_1_T.setElement(uk1,rad_ker,sl_ker,(kernely_full_1_T.getElement(uk1,rad_ker,sl_ker) - alfa * 0.01*grad_T.getElement(grad_depth,rad_v2,sl_v2)*vrstva0_T.getElement(0,(rad_v2*vrstva2_T.getRows()+rad_v1+rad_ker),(sl_v2*vrstva2_T.getCols()+sl_v1+sl_ker))));
                                            bias_full_k1_T[uk1] = bias_full_k1_T[uk1] - alfa * 0.01*grad_T.getElement(grad_depth,rad_v2,sl_v2);
                                        }else{
                                            kernely_full_1_T.setElement(uk1,rad_ker,sl_ker,(kernely_full_1_T.getElement(uk1,rad_ker,sl_ker) - alfa*grad_T.getElement(grad_depth,rad_v2,sl_v2)*vrstva0_T.getElement(0,(rad_v2*vrstva2_T.getRows()+rad_v1+rad_ker),(sl_v2*vrstva2_T.getCols()+sl_v1+sl_ker))));
                                            bias_full_k1_T[uk1] = bias_full_k1_T[uk1] - alfa*grad_T.getElement(grad_depth,rad_v2,sl_v2);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    }
}
}
}
}

void NN::cnn_full_val(int velic){
    vystupy.clear();
if(velic == 2){ 
    if (Q_val_vstup.size() < 36) {
        std::cout << "Delka Q_val_vstup musi byt vetsi nez 36";
        exit(0);
    }
    ////////////////////////////////KONVOLUCE
    Tenzor<double> vrstva0_Q;
    Tenzor<double> vrstva1_Q;
    Tenzor<double> vrstva2_Q;
    Tenzor<double> vrstva_final_Q;
    Matice<double> dataprocnn_Q;
    std::vector<double> current_kus_Q;

    Tenzor<double> vrstva0_R;
    Tenzor<double> vrstva1_R;
    Tenzor<double> vrstva2_R;
    Tenzor<double> vrstva_final_R;
    Matice<double> dataprocnn_R;
    std::vector<double> current_kus_R;

    std::vector<double> vystzkonv;

    for (int kroky = 0; kroky < ((Q_val_vstup.size()-36));kroky++){
        vrstva0_Q.resize(0,0,0);
        vrstva1_Q.resize(0,0,0);
        vrstva2_Q.resize(0,0,0);
        vrstva_final_Q.resize(0,0,0);

        for(int kus = 0; kus < 36; kus++){
            current_kus_Q.push_back(Q_val_vstup[kroky + kus]);
        }

        dataprocnn_Q = udelej_radky(6,current_kus_Q);
        current_kus_Q.clear();
        vrstva0_Q.add_matrix(dataprocnn_Q);
        dataprocnn_Q.resize(0,0);
        vrstva1_Q = konvo_3d(vrstva0_Q,kernely_full_1_Q);
        for(int dep_v1 = 0;dep_v1<vrstva1_Q.getDepth();dep_v1++){
            for(int row_v1 = 0;row_v1<vrstva1_Q.getRows();row_v1++){
                for(int col_v1 = 0;col_v1<vrstva1_Q.getCols();col_v1++){
                    vrstva1_Q.setElement(dep_v1,row_v1,col_v1,(vrstva1_Q.getElement(dep_v1,row_v1,col_v1)+bias_full_k1_Q[dep_v1]));
                    if(vrstva1_Q.getElement(dep_v1,row_v1,col_v1)<0.0){
                        vrstva1_Q.setElement(dep_v1,row_v1,col_v1,(vrstva1_Q.getElement(dep_v1,row_v1,col_v1)*0.01));
                    }
                }
            }
        }
        vrstva2_Q = max_pool_fullstep_3d(vrstva1_Q,2,2);
        vrstva_final_Q = konvo_3d(vrstva2_Q,kernely_full_2_Q);
        for(int dep_v2=0;dep_v2<vrstva2_Q.getDepth();dep_v2++){
            for(int dep_k2=0;dep_k2<kernely_full_2_Q.getDepth();dep_k2++){
                vrstva_final_Q.setElement((dep_v2*kernely_full_2_Q.getDepth()+dep_k2),0,0,(vrstva_final_Q.getElement((dep_v2*kernely_full_2_Q.getDepth()+dep_k2),0,0)+bias_full_k2_Q[dep_k2]));
                if(vrstva_final_Q.getElement((dep_v2*kernely_full_2_Q.getDepth()+dep_k2),0,0)<0.0){
                    vrstva_final_Q.setElement((dep_v2*kernely_full_2_Q.getDepth()+dep_k2),0,0,(vrstva_final_Q.getElement((dep_v2*kernely_full_2_Q.getDepth()+dep_k2),0,0)*0.01));
                }
            }
        }


        vrstva0_R.resize(0,0,0);
        vrstva1_R.resize(0,0,0);
        vrstva2_R.resize(0,0,0);
        vrstva_final_R.resize(0,0,0);

        for(int kus = 0; kus < 36; kus++){
            current_kus_R.push_back(R_val_vstup[kroky + kus]);
        }

        dataprocnn_R = udelej_radky(6,current_kus_R);
        current_kus_R.clear();
        vrstva0_R.add_matrix(dataprocnn_R);
        dataprocnn_R.resize(0,0);
        vrstva1_R = konvo_3d(vrstva0_R,kernely_full_1_R);
        for(int dep_v1 = 0;dep_v1<vrstva1_R.getDepth();dep_v1++){
            for(int row_v1 = 0;row_v1<vrstva1_R.getRows();row_v1++){
                for(int col_v1 = 0;col_v1<vrstva1_R.getCols();col_v1++){
                    vrstva1_R.setElement(dep_v1,row_v1,col_v1,(vrstva1_R.getElement(dep_v1,row_v1,col_v1)+bias_full_k1_R[dep_v1]));
                    if(vrstva1_R.getElement(dep_v1,row_v1,col_v1)<0.0){
                        vrstva1_R.setElement(dep_v1,row_v1,col_v1,(vrstva1_R.getElement(dep_v1,row_v1,col_v1)*0.01));
                    }
                }
            }
        }
        vrstva2_R = max_pool_fullstep_3d(vrstva1_R,2,2);
        vrstva_final_R = konvo_3d(vrstva2_R,kernely_full_2_R);
        for(int dep_v2=0;dep_v2<vrstva2_R.getDepth();dep_v2++){
            for(int dep_k2=0;dep_k2<kernely_full_2_R.getDepth();dep_k2++){
                vrstva_final_R.setElement((dep_v2*kernely_full_2_R.getDepth()+dep_k2),0,0,(vrstva_final_R.getElement((dep_v2*kernely_full_2_R.getDepth()+dep_k2),0,0)+bias_full_k2_R[dep_k2]));
                if(vrstva_final_R.getElement((dep_v2*kernely_full_2_R.getDepth()+dep_k2),0,0)<0.0){
                    vrstva_final_R.setElement((dep_v2*kernely_full_2_R.getDepth()+dep_k2),0,0,(vrstva_final_R.getElement((dep_v2*kernely_full_2_R.getDepth()+dep_k2),0,0)*0.01));
                }
            }
        }

    ///////////////////////////MLP
    vystzkonv.clear();

    for(int i = 0;i<vrstva_final_Q.getDepth();++i){
        vystzkonv.push_back(vrstva_final_Q.getElement(i,0,0));
    }

    for(int i = 0;i<vrstva_final_R.getDepth();++i){
        vystzkonv.push_back(vrstva_final_R.getElement(i,0,0));
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
            for (int i = 0; i<pom_vystup.size();i++){  
                vystupy.push_back(pom_vystup[i]);
            }
    }
}else if(velic == 3){ 
    if (Q_val_vstup.size() < 36) {
        std::cout << "Delka Q_val_vstup musi byt vetsi nez 36";
        exit(0);
    }

    ////////////////////////////////KONVOLUCE
    Tenzor<double> vrstva0_Q;
    Tenzor<double> vrstva1_Q;
    Tenzor<double> vrstva2_Q;
    Tenzor<double> vrstva_final_Q;
    Matice<double> dataprocnn_Q;
    std::vector<double> current_kus_Q;

    Tenzor<double> vrstva0_R;
    Tenzor<double> vrstva1_R;
    Tenzor<double> vrstva2_R;
    Tenzor<double> vrstva_final_R;
    Matice<double> dataprocnn_R;
    std::vector<double> current_kus_R;

    Tenzor<double> vrstva0_T;
    Tenzor<double> vrstva1_T;
    Tenzor<double> vrstva2_T;
    Tenzor<double> vrstva_final_T;
    Matice<double> dataprocnn_T;
    std::vector<double> current_kus_T;

    std::vector<double> vystzkonv;

    for (int kroky = 0; kroky < (Q_val_vstup.size()-36);kroky++){
        vrstva0_Q.resize(0,0,0);
        vrstva1_Q.resize(0,0,0);
        vrstva2_Q.resize(0,0,0);
        vrstva_final_Q.resize(0,0,0);

        for(int kus = 0; kus < 36; kus++){
            current_kus_Q.push_back(Q_val_vstup[kroky + kus]);
        }

        dataprocnn_Q = udelej_radky(6,current_kus_Q);
        current_kus_Q.clear();
        vrstva0_Q.add_matrix(dataprocnn_Q);
        dataprocnn_Q.resize(0,0);
        vrstva1_Q = konvo_3d(vrstva0_Q,kernely_full_1_Q);
        for(int dep_v1 = 0;dep_v1<vrstva1_Q.getDepth();dep_v1++){
            for(int row_v1 = 0;row_v1<vrstva1_Q.getRows();row_v1++){
                for(int col_v1 = 0;col_v1<vrstva1_Q.getCols();col_v1++){
                    vrstva1_Q.setElement(dep_v1,row_v1,col_v1,(vrstva1_Q.getElement(dep_v1,row_v1,col_v1)+bias_full_k1_Q[dep_v1]));
                    if(vrstva1_Q.getElement(dep_v1,row_v1,col_v1)<0.0){
                        vrstva1_Q.setElement(dep_v1,row_v1,col_v1,(vrstva1_Q.getElement(dep_v1,row_v1,col_v1)*0.01));
                    }
                }
            }
        }
        vrstva2_Q = max_pool_fullstep_3d(vrstva1_Q,2,2);
        vrstva_final_Q = konvo_3d(vrstva2_Q,kernely_full_2_Q);
        for(int dep_v2=0;dep_v2<vrstva2_Q.getDepth();dep_v2++){
            for(int dep_k2=0;dep_k2<kernely_full_2_Q.getDepth();dep_k2++){
                vrstva_final_Q.setElement((dep_v2*kernely_full_2_Q.getDepth()+dep_k2),0,0,(vrstva_final_Q.getElement((dep_v2*kernely_full_2_Q.getDepth()+dep_k2),0,0)+bias_full_k2_Q[dep_k2]));
                if(vrstva_final_Q.getElement((dep_v2*kernely_full_2_Q.getDepth()+dep_k2),0,0)<0.0){
                    vrstva_final_Q.setElement((dep_v2*kernely_full_2_Q.getDepth()+dep_k2),0,0,(vrstva_final_Q.getElement((dep_v2*kernely_full_2_Q.getDepth()+dep_k2),0,0)*0.01));
                }
            }
        }


        vrstva0_R.resize(0,0,0);
        vrstva1_R.resize(0,0,0);
        vrstva2_R.resize(0,0,0);
        vrstva_final_R.resize(0,0,0);

        for(int kus = 0; kus < 36; kus++){
            current_kus_R.push_back(R_val_vstup[kroky + kus]);
        }

        dataprocnn_R = udelej_radky(6,current_kus_R);
        current_kus_R.clear();
        vrstva0_R.add_matrix(dataprocnn_R);
        dataprocnn_R.resize(0,0);
        vrstva1_R = konvo_3d(vrstva0_R,kernely_full_1_R);
        for(int dep_v1 = 0;dep_v1<vrstva1_R.getDepth();dep_v1++){
            for(int row_v1 = 0;row_v1<vrstva1_R.getRows();row_v1++){
                for(int col_v1 = 0;col_v1<vrstva1_R.getCols();col_v1++){
                    vrstva1_R.setElement(dep_v1,row_v1,col_v1,(vrstva1_R.getElement(dep_v1,row_v1,col_v1)+bias_full_k1_R[dep_v1]));
                    if(vrstva1_R.getElement(dep_v1,row_v1,col_v1)<0.0){
                        vrstva1_R.setElement(dep_v1,row_v1,col_v1,(vrstva1_R.getElement(dep_v1,row_v1,col_v1)*0.01));
                    }
                }
            }
        }
        vrstva2_R = max_pool_fullstep_3d(vrstva1_R,2,2);
        vrstva_final_R = konvo_3d(vrstva2_R,kernely_full_2_R);
        for(int dep_v2=0;dep_v2<vrstva2_R.getDepth();dep_v2++){
            for(int dep_k2=0;dep_k2<kernely_full_2_R.getDepth();dep_k2++){
                vrstva_final_R.setElement((dep_v2*kernely_full_2_R.getDepth()+dep_k2),0,0,(vrstva_final_R.getElement((dep_v2*kernely_full_2_R.getDepth()+dep_k2),0,0)+bias_full_k2_R[dep_k2]));
                if(vrstva_final_R.getElement((dep_v2*kernely_full_2_R.getDepth()+dep_k2),0,0)<0.0){
                    vrstva_final_R.setElement((dep_v2*kernely_full_2_R.getDepth()+dep_k2),0,0,(vrstva_final_R.getElement((dep_v2*kernely_full_2_R.getDepth()+dep_k2),0,0)*0.01));
                }
            }
        }

        vrstva0_T.resize(0,0,0);
        vrstva1_T.resize(0,0,0);
        vrstva2_T.resize(0,0,0);
        vrstva_final_T.resize(0,0,0);

        for(int kus = 0; kus < 36; kus++){
            current_kus_T.push_back(T_val_vstup[kroky + kus]);
        }

        dataprocnn_T = udelej_radky(6,current_kus_T);
        current_kus_T.clear();
        vrstva0_T.add_matrix(dataprocnn_T);
        dataprocnn_T.resize(0,0);
        vrstva1_T = konvo_3d(vrstva0_T,kernely_full_1_T);
        for(int dep_v1 = 0;dep_v1<vrstva1_T.getDepth();dep_v1++){
            for(int row_v1 = 0;row_v1<vrstva1_T.getRows();row_v1++){
                for(int col_v1 = 0;col_v1<vrstva1_T.getCols();col_v1++){
                    vrstva1_T.setElement(dep_v1,row_v1,col_v1,(vrstva1_T.getElement(dep_v1,row_v1,col_v1)+bias_full_k1_T[dep_v1]));
                    if(vrstva1_T.getElement(dep_v1,row_v1,col_v1)<0.0){
                        vrstva1_T.setElement(dep_v1,row_v1,col_v1,(vrstva1_T.getElement(dep_v1,row_v1,col_v1)*0.01));
                    }
                }
            }
        }
        vrstva2_T = max_pool_fullstep_3d(vrstva1_T,2,2);
        vrstva_final_T = konvo_3d(vrstva2_T,kernely_full_2_T);
        for(int dep_v2=0;dep_v2<vrstva2_T.getDepth();dep_v2++){
            for(int dep_k2=0;dep_k2<kernely_full_2_T.getDepth();dep_k2++){
                vrstva_final_T.setElement((dep_v2*kernely_full_2_T.getDepth()+dep_k2),0,0,(vrstva_final_T.getElement((dep_v2*kernely_full_2_T.getDepth()+dep_k2),0,0)+bias_full_k2_T[dep_k2]));
                if(vrstva_final_T.getElement((dep_v2*kernely_full_2_T.getDepth()+dep_k2),0,0)<0.0){
                    vrstva_final_T.setElement((dep_v2*kernely_full_2_T.getDepth()+dep_k2),0,0,(vrstva_final_T.getElement((dep_v2*kernely_full_2_T.getDepth()+dep_k2),0,0)*0.01));
                }
            }
        }

    ///////////////////////////MLP
    vystzkonv.clear();

    for(int i = 0;i<vrstva_final_Q.getDepth();++i){
        vystzkonv.push_back(vrstva_final_Q.getElement(i,0,0));
    }

    for(int i = 0;i<vrstva_final_R.getDepth();++i){
        vystzkonv.push_back(vrstva_final_R.getElement(i,0,0));
    }

    for(int i = 0;i<vrstva_final_T.getDepth();++i){
        vystzkonv.push_back(vrstva_final_T.getElement(i,0,0));
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
            for (int i = 0; i<pom_vystup.size();i++){  
                vystupy.push_back(pom_vystup[i]);
            }
    }
}
}

void NN::cnn1D_cal(size_t vel_ker, size_t poc_ker, int iter, int velic){

if(velic == 3){
    kernely_1D.resize(0,0,0);
    Matice<double>ker_in_Q(poc_ker,vel_ker);
    Matice<double>ker_in_R(poc_ker,vel_ker);
    Matice<double>ker_in_T(poc_ker,vel_ker);
    ker_in_Q.rand_vypln(0.0,0.1);
    ker_in_R.rand_vypln(0.0,0.01);
    ker_in_T.rand_vypln(0.0,0.001);
    kernely_1D.add_matrix(ker_in_Q);
    kernely_1D.add_matrix(ker_in_R);
    kernely_1D.add_matrix(ker_in_T);
    ker_in_Q.resize(0,0);
    ker_in_R.resize(0,0);
    ker_in_T.resize(0,0);
    biaskonv_1D.resize(3,poc_ker);
    double deltazmlp;
    Tenzor<double> uprava_k(3,poc_ker,vel_ker);
    std::vector<double> vystzkonv;   
    
    if (Q_kal_vstup.size() != R_kal_vstup.size()|| Q_kal_vstup.size() != T_kal_vstup.size()) {
        std::cout << "vstupni řady nejsou stejně dlouhý";
        exit(0);
    }

//////////////////////////////////////////////////////////// KALIBRACE //////////////////////////////////////////////////////
    for(int m = 0; m < iter; m++){
        for(int kroky = 0; kroky < (Q_kal_vstup.size() - vel_ker); kroky++){

//// KONVOLUCE //////////////////////////////////////////////////////////////////////////////////////////////////////////////                
                vystzkonv.clear();
    //if(R_kal_vstup[kroky+vel_ker-1]>0.0 & R_kal_vstup[kroky+vel_ker-2]>0.0 & R_kal_vstup[kroky+vel_ker-3]>0.0){
                for(int i = 0; i < poc_ker; i++){
                    double konvo = 0.0;
                    for(int j = 0; j < vel_ker; j++){
                        konvo += Q_kal_vstup[kroky+j] * kernely_1D.getElement(0,i,j);
                    }
                    konvo += biaskonv_1D.getElement(0,i);
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
                    konvo += biaskonv_1D.getElement(1,i);
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
                    konvo += biaskonv_1D.getElement(2,i);
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
            for(int k = 0; k < sit[i][j].vahy.size();++k)
                sit[i][j].vahy[k] = sit[i][j].vahy[k] - alfa * sit[i][j].delta * sit[i][j].vstupy[k];
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
                                biaskonv_1D.setElement(i,j,(biaskonv_1D.getElement(i,j) - alfa * deltazmlp));
                            } else{
                                for(int k = 0; k < vel_ker; k++){
                                    kernely_1D.setElement(i,j,k,(kernely_1D.getElement(i,j,k) - alfa * 0.01 * uprava_k.getElement(i,j,k)));
                                }
                                biaskonv_1D.setElement(i,j,(biaskonv_1D.getElement(i,j) - alfa * 0.01 * deltazmlp));
                            }

                        }
                    }
                }
        }
    }          
        }else if(velic == 2){
            kernely_1D.resize(0,0,0);
            Matice<double>ker_in_Q(poc_ker,vel_ker);
            Matice<double>ker_in_R(poc_ker,vel_ker);
            ker_in_Q.rand_vypln(0.0,0.1);
            ker_in_R.rand_vypln(0.0,0.1);
            kernely_1D.add_matrix(ker_in_Q);
            kernely_1D.add_matrix(ker_in_R);
            ker_in_Q.resize(0,0);
            ker_in_R.resize(0,0);
            biaskonv_1D.resize(2,poc_ker);
            double deltazmlp;
            Tenzor<double> uprava_k(2,poc_ker,vel_ker);
            std::vector<double> vystzkonv;   
            
            if (Q_kal_vstup.size() != R_kal_vstup.size()) {
                std::cout << "vstupni řady nejsou stejně dlouhý";
                exit(0);
            }
        
        //////////////////////////////////////////////////////////// KALIBRACE //////////////////////////////////////////////////////
            for(int m = 0; m < iter; m++){
                for(int kroky = 0; kroky < (Q_kal_vstup.size() - vel_ker); kroky++){
        
        //// KONVOLUCE //////////////////////////////////////////////////////////////////////////////////////////////////////////////                
                        vystzkonv.clear();
 //if(R_kal_vstup[kroky+vel_ker-1]>0.0 & R_kal_vstup[kroky+vel_ker-2]>0.0 & R_kal_vstup[kroky+vel_ker-3]>0.0){
                        for(int i = 0; i < poc_ker; i++){
                            double konvo = 0.0;
                            for(int j = 0; j < vel_ker; j++){
                                konvo += Q_kal_vstup[kroky+j] * kernely_1D.getElement(0,i,j);
                            }
                            konvo += biaskonv_1D.getElement(0,i);
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
                            konvo += biaskonv_1D.getElement(1,i);
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
                    for(int k = 0; k < sit[i][j].vahy.size();++k)
                        sit[i][j].vahy[k] = sit[i][j].vahy[k] - alfa * sit[i][j].delta * sit[i][j].vstupy[k];
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
        
                            for(int i = 0; i < 2; i++){
                                for(int j = 0; j < poc_ker; j++){
                                    if(vystzkonv[i*poc_ker+j] > 0.0){
                                        for(int k = 0; k < vel_ker; k++){
                                            kernely_1D.setElement(i,j,k,(kernely_1D.getElement(i,j,k) - alfa * uprava_k.getElement(i,j,k)));
                                        }
                                        biaskonv_1D.setElement(i,j,(biaskonv_1D.getElement(i,j) - alfa * deltazmlp));
                                    } else{
                                        for(int k = 0; k < vel_ker; k++){
                                            kernely_1D.setElement(i,j,k,(kernely_1D.getElement(i,j,k) - alfa * 0.01 * uprava_k.getElement(i,j,k)));
                                        }
                                        biaskonv_1D.setElement(i,j,(biaskonv_1D.getElement(i,j) - alfa * 0.01 * deltazmlp));
                                    }
        
                                }
                            }
                        }
                }
            }
        }else {
    kernely_1D.resize(1,poc_ker,vel_ker);
    kernely_1D.rand_vypln(0.0,0.1);
    biaskonv_1D.resize(1,poc_ker);
    double deltazmlp;
    Tenzor<double> uprava_k(1,poc_ker,vel_ker);
    std::vector<double> vystzkonv;   


//////////////////////////////////////////////////////////// KALIBRACE //////////////////////////////////////////////////////
    for(int m = 0; m < iter; m++){
        for(int kroky = 0; kroky < (Q_kal_vstup.size() - vel_ker); kroky++){

//// KONVOLUCE //////////////////////////////////////////////////////////////////////////////////////////////////////////////                
                vystzkonv.clear();

                for(int i = 0; i < poc_ker; i++){
                    double konvo = 0.0;
                    for(int j = 0; j < vel_ker; j++){
                        konvo += Q_kal_vstup[kroky+j] * kernely_1D.getElement(0,i,j);
                    }
                    konvo += biaskonv_1D.getElement(0,i);
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
            for(int k = 0; k < sit[i][j].vahy.size();++k)
                sit[i][j].vahy[k] = sit[i][j].vahy[k] - alfa * sit[i][j].delta * sit[i][j].vstupy[k];
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

                    for(int i = 0; i < 1; i++){
                        for(int j = 0; j < poc_ker; j++){
                            if(vystzkonv[i*poc_ker+j] > 0.0){
                                for(int k = 0; k < vel_ker; k++){
                                    kernely_1D.setElement(i,j,k,(kernely_1D.getElement(i,j,k) - alfa * uprava_k.getElement(i,j,k)));
                                }
                                biaskonv_1D.setElement(i,j,(biaskonv_1D.getElement(i,j) - alfa * deltazmlp));
                            } else{
                                for(int k = 0; k < vel_ker; k++){
                                    kernely_1D.setElement(i,j,k,(kernely_1D.getElement(i,j,k) - alfa * 0.01 * uprava_k.getElement(i,j,k)));
                                }
                                biaskonv_1D.setElement(i,j,(biaskonv_1D.getElement(i,j) - alfa * 0.01 * deltazmlp));
                            }

                        }
                    }
                }
        }
    }        
        }
}
void NN::cnn1D_val(int velic){
    std::vector<double> vystzkonv;   
if(velic == 3){
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
                    konvo += biaskonv_1D.getElement(0,i);
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
                    konvo += biaskonv_1D.getElement(1,i);
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
                    konvo += biaskonv_1D.getElement(2,i);
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
}else if(velic == 2){
    if (Q_val_vstup.size() != R_val_vstup.size()) {
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
                    konvo += biaskonv_1D.getElement(0,i);
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
                    konvo += biaskonv_1D.getElement(1,i);
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
}else{
    vystupy.clear();
    for(int kroky = 0; kroky < (Q_val_vstup.size() - kernely_1D.getCols()); kroky++){

//// KONVOLUCE //////////////////////////////////////////////////////////////////////////////////////////////////////////////                
                vystzkonv.clear();

                for(int i = 0; i < kernely_1D.getRows(); i++){
                    double konvo = 0.0;
                    for(int j = 0; j < kernely_1D.getCols(); j++){
                        konvo += Q_val_vstup[kroky+j] * kernely_1D.getElement(0,i,j);
                    }
                    konvo += biaskonv_1D.getElement(0,i);
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
}

void NN::cnn1Dreal_cal(int iter, int velic){
    alfa = 0.0001;
    kernely_1Dreal_1.clear();
    kernely_1Dreal_2.clear();
    kernely_1Dreal_3.clear();

    bias_1Dreal_1.clear();
    bias_1Dreal_2.clear();
    bias_1Dreal_3.clear();

    std::vector<double>init;
    for (int i = 0; i<5;i++){
        for(int j = 0; j<15;j++){
            init.push_back(random(0.0,0.1));
        }
        kernely_1Dreal_1.push_back(init);
        init.clear();

        bias_1Dreal_1.push_back(0.0);
    }

    for (int i = 0; i<2;i++){
        for(int j = 0; j<10;j++){
            init.push_back(random(0.0,0.1));
        }
        kernely_1Dreal_2.push_back(init);
        init.clear();

        for(int j = 0; j<7;j++){
            init.push_back(random(0.0,0.1));
        }
        kernely_1Dreal_3.push_back(init);
        init.clear();

        bias_1Dreal_2.push_back(0.0);
        bias_1Dreal_3.push_back(0.0);
    }
    std::vector<double> vystzkonv;
    std::vector<std::vector<double>>vrstva1;
    std::vector<std::vector<double>>vrstva2;
    std::vector<std::vector<double>>delta_1_2;
    std::vector<std::vector<double>>delta_2_3;
    double deltazmlp; 
    if(velic == 2){
        if (Q_kal_vstup.size() != R_kal_vstup.size()) {
            std::cout << "vstupni řady nejsou stejně dlouhý";
            exit(0);
        }
    
    //////////////////////////////////////////////////////////// KALIBRACE //////////////////////////////////////////////////////
        for(int iters = 0; iters < iter; iters++){
            for(int kroky = 0; kroky < (Q_kal_vstup.size() - 30); kroky++){
                vystzkonv.clear();
                vrstva1.clear();
                vrstva2.clear();
                delta_1_2.clear();
                delta_2_3.clear();
                chtenejout.clear();
                chtenejout.push_back(Q_kal_vstup[kroky+30]);
                for(int i = 0; i<5;i++){
                    for(int j = 0; j<16;j++){
                        double konvo = 0.0;
                        for(int k = 0; k<15;k++){
                            konvo += kernely_1Dreal_1[i][k] * Q_kal_vstup[kroky +j+k];
                        }
                        konvo+=bias_1Dreal_1[i];
                        if(konvo<0.0){
                            init.push_back(konvo * 0.01);
                        }else{
                            init.push_back(konvo * 0.01);
                        }
                    }
                    vrstva1.push_back(init);
                    init.clear();
                }

                for(int i = 0; i<5;i++){
                    for(int j = 0; j<16;j++){
                        double konvo = 0.0;
                        for(int k = 0; k<15;k++){
                            konvo += kernely_1Dreal_1[i][k] * R_kal_vstup[kroky +j+k];
                        }
                        konvo+=bias_1Dreal_1[i];
                        if(konvo<0.0){
                            init.push_back(konvo * 0.01);
                        }else{
                            init.push_back(konvo);
                        }
                    }
                    vrstva1.push_back(init);
                    init.clear();
                }
                

                for(int i = 0; i<2;i++){
                    for(int j = 0; j<10; j++){
                        for(int k = 0;k<7;k++){
                            double konvo = 0.0;
                            for(int l = 0; l<10;l++){
                                konvo+=kernely_1Dreal_2[i][l]*vrstva1[j][k+l];
                            }
                            konvo+=bias_1Dreal_2[i];
                            if(konvo<0.0){
                                init.push_back(konvo * 0.01);
                            }else{
                                init.push_back(konvo);
                            }
                        }
                        vrstva2.push_back(init);
                        init.clear();
                    }
                }

                for(int i = 0; i<2;i++){
                    for(int j = 0; j<20; j++){
                            double konvo = 0.0;
                            for(int l = 0; l<7;l++){
                                konvo+=kernely_1Dreal_3[i][l]*vrstva2[j][l];
                            }
                            konvo+=bias_1Dreal_3[i];
                            if(konvo<0.0){
                                vystzkonv.push_back(konvo * 0.01);
                            }else{
                                vystzkonv.push_back(konvo);
                            }
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
                sit[pocet_vrstev-1][rozmery[pocet_vrstev-1]-1].delta = pom_vystup[0] - chtenejout[0];

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

    ///////////////////////// CNN BACKPROP ////////////////////////////////////////////////////////////////////////////////////
    for (int neur = 0;neur<rozmery[0];++neur){
        deltazmlp = sit[0][neur].delta;

        for(int i = 0; i<2;i++){
            for(int j = 0; j<7;j++){
                init.push_back(kernely_1Dreal_3[i][6-j] * deltazmlp);
            }
            delta_2_3.push_back(init);
            init.clear();
        }

        std::vector<double>obracfiltr;
        for(int i = 0; i<2;i++){
            for(int j = 0; j<2;j++){
                for(int k=0;k<10;k++){
                    obracfiltr.push_back(kernely_1Dreal_2[i][9-k]);
                }
                for(int k=0;k<6;k++){
                    obracfiltr.push_back(0.0);
                    obracfiltr.insert(obracfiltr.begin(), 0.0);
                }
                for(int k = 0;k<16;k++){
                    double konvo = 0.0;
                    for(int l = 0;l<7;l++){
                        konvo+=obracfiltr[k+l]*delta_2_3[j][l];
                    }
                    init.push_back(konvo);
                }
                delta_1_2.push_back(init);
                obracfiltr.clear();
                init.clear();
            }
        }

        for(int i = 0; i<2;i++){
            for(int j = 0; j<20;j++){
                if(vystzkonv[i*20+j]<0.0){    
                    for(int k = 0; k<7;k++){
                        kernely_1Dreal_3[i][j] -= 0.01 * alfa * vrstva2[j][k]*deltazmlp;
                    }
                    bias_1Dreal_3[i]-=0.01 * alfa * deltazmlp;
                }else{
                    for(int k = 0; k<7;k++){
                        kernely_1Dreal_3[i][j] -= alfa * vrstva2[j][k]*deltazmlp;
                    }
                    bias_1Dreal_3[i] -= alfa * deltazmlp;
                }
            }
        }

        for(int i = 0; i<2;i++){
            for(int j = 0; j<2;j++){
                for(int k = 0;k<10;k++){
                    for(int m = 0;m<7;m++){
                        for (int l = 0;l<10;l++){
                            if(vrstva2[i*10+k][m]<0.0){
                                kernely_1Dreal_2[i][l]-= 0.01 * alfa * vrstva1[k][l+m] * delta_2_3[j][m];
                            }else{
                                kernely_1Dreal_2[i][l]-= alfa * vrstva1[k][l+m] * delta_2_3[j][m];
                            }
                        }
                        if(vrstva2[i*10+k][m]<0.0){
                            bias_1Dreal_2[i] -= 0.01 * alfa * delta_2_3[j][m];
                        }else{
                            bias_1Dreal_2[i] -= alfa * delta_2_3[j][m];
                        }
                    }
                }
            }
        }

        for(int i = 0; i<5;i++){
            for(int j = 0; j<4;j++){
                    for(int m = 0;m<16;m++){
                        for (int l = 0;l<15;l++){
                            if(vrstva1[i][m]<0.0){
                                kernely_1Dreal_1[i][l]-= 0.01 * alfa * Q_kal_vstup[kroky +l+m] * delta_1_2[j][m];
                            }else{
                                kernely_1Dreal_1[i][l]-= alfa * Q_kal_vstup[kroky +l+m] * delta_1_2[j][m];
                            }
                        }
                        if(vrstva1[i][m]<0.0){
                            bias_1Dreal_1[i] -= 0.01 * alfa * delta_1_2[j][m];
                        }else{
                            bias_1Dreal_1[i] -= alfa * delta_1_2[j][m];
                        }
                    }
            
            }
        }

        for(int i = 0; i<5;i++){
            for(int j = 0; j<4;j++){
                    for(int m = 0;m<16;m++){
                        for (int l = 0;l<15;l++){
                            if(vrstva1[5+i][m]<0.0){
                                kernely_1Dreal_1[i][l]-= 0.01 * alfa * R_kal_vstup[kroky +l+m] * delta_1_2[j][m];
                            }else{
                                kernely_1Dreal_1[i][l]-= alfa * R_kal_vstup[kroky +l+m] * delta_1_2[j][m];
                            }
                        }
                        if(vrstva1[i][m]<0.0){
                            bias_1Dreal_1[i] -= 0.01 * alfa * delta_1_2[j][m];
                        }else{
                            bias_1Dreal_1[i] -= alfa * delta_1_2[j][m];
                        }
                    }
            
            }
        }   
            }

        

        }
    }
    } else if(velic == 3){
        if (Q_kal_vstup.size() != R_kal_vstup.size() || Q_kal_vstup.size() != T_kal_vstup.size()) {
            std::cout << "vstupni řady nejsou stejně dlouhý";
            exit(0);
        }
    
    //////////////////////////////////////////////////////////// KALIBRACE //////////////////////////////////////////////////////
        for(int iters = 0; iters < iter; iters++){
            for(int kroky = 0; kroky < (Q_kal_vstup.size() - 30); kroky++){
                vystzkonv.clear();
                vrstva1.clear();
                vrstva2.clear();
                delta_1_2.clear();
                delta_2_3.clear();
                chtenejout.clear();
                chtenejout.push_back(Q_kal_vstup[kroky+30]);
                for(int i = 0; i<5;i++){
                    for(int j = 0; j<16;j++){
                        double konvo = 0.0;
                        for(int k = 0; k<15;k++){
                            konvo += kernely_1Dreal_1[i][k] * Q_kal_vstup[kroky +j+k];
                        }
                        konvo+=bias_1Dreal_1[i];
                        if(konvo<0.0){
                            init.push_back(konvo * 0.01);
                        }else{
                            init.push_back(konvo);
                        }
                    }
                    vrstva1.push_back(init);
                    init.clear();
                }

                for(int i = 0; i<5;i++){
                    for(int j = 0; j<16;j++){
                        double konvo = 0.0;
                        for(int k = 0; k<15;k++){
                            konvo += kernely_1Dreal_1[i][k] * R_kal_vstup[kroky +j+k];
                        }
                        konvo+=bias_1Dreal_1[i];
                        if(konvo<0.0){
                            init.push_back(konvo * 0.01);
                        }else{
                            init.push_back(konvo);
                        }
                    }
                    vrstva1.push_back(init);
                    init.clear();
                }

                for(int i = 0; i<5;i++){
                    for(int j = 0; j<16;j++){
                        double konvo = 0.0;
                        for(int k = 0; k<15;k++){
                            konvo += kernely_1Dreal_1[i][k] * T_kal_vstup[kroky +j+k];
                        }
                        konvo+=bias_1Dreal_1[i];
                        if(konvo<0.0){
                            init.push_back(konvo * 0.01);
                        }else{
                            init.push_back(konvo);
                        }
                    }
                    vrstva1.push_back(init);
                    init.clear();
                }

                for(int i = 0; i<2;i++){
                    for(int j = 0; j<15; j++){
                        for(int k = 0;k<7;k++){
                            double konvo = 0.0;
                            for(int l = 0; l<10;l++){
                                konvo+=kernely_1Dreal_2[i][l]*vrstva1[j][k+l];
                            }
                            konvo+=bias_1Dreal_2[i];
                            if(konvo<0.0){
                                init.push_back(konvo * 0.01);
                            }else{
                                init.push_back(konvo);
                            }
                        }
                        vrstva2.push_back(init);
                        init.clear();
                    }
                }

                for(int i = 0; i<2;i++){
                    for(int j = 0; j<30; j++){
                            double konvo = 0.0;
                            for(int l = 0; l<7;l++){
                                konvo+=kernely_1Dreal_3[i][l]*vrstva2[j][l];
                            }
                            konvo+=bias_1Dreal_3[i];
                            if(konvo<0.0){
                                vystzkonv.push_back(konvo * 0.01);
                            }else{
                                vystzkonv.push_back(konvo);
                            }
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
                sit[pocet_vrstev-1][rozmery[pocet_vrstev-1]-1].delta = pom_vystup[0] - chtenejout[0];

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
///////////////////////// CNN BACKPROP ////////////////////////////////////////////////////////////////////////////////////
    for (int neur = 0;neur<rozmery[0];++neur){
        deltazmlp = sit[0][neur].delta;

        for(int i = 0; i<2;i++){
            for(int j = 0; j<7;j++){
                init.push_back(kernely_1Dreal_3[i][6-j] * deltazmlp);
            }
            delta_2_3.push_back(init);
            init.clear();
        }

        std::vector<double>obracfiltr;
        for(int i = 0; i<2;i++){
            for(int j = 0; j<2;j++){
                for(int k=0;k<10;k++){
                    obracfiltr.push_back(kernely_1Dreal_2[i][9-k]);
                }
                for(int k=0;k<6;k++){
                    obracfiltr.push_back(0.0);
                    obracfiltr.insert(obracfiltr.begin(), 0.0);
                }
                for(int k = 0;k<16;k++){
                    double konvo = 0.0;
                    for(int l = 0;l<7;l++){
                        konvo+=obracfiltr[k+l]*delta_2_3[j][l];
                    }
                    init.push_back(konvo);
                }
                delta_1_2.push_back(init);
                obracfiltr.clear();
                init.clear();
            }
        }

        for(int i = 0; i<2;i++){
            for(int j = 0; j<30;j++){
                if(vystzkonv[i*30+j]<0.0){    
                    for(int k = 0; k<7;k++){
                        kernely_1Dreal_3[i][j] -= 0.01 * alfa * vrstva2[j][k]*deltazmlp;
                    }
                    bias_1Dreal_3[i]-=0.01 * alfa * deltazmlp;
                }else{
                    for(int k = 0; k<7;k++){
                        kernely_1Dreal_3[i][j] -= alfa * vrstva2[j][k]*deltazmlp;
                    }
                    bias_1Dreal_3[i] -= alfa * deltazmlp;
                }
            }
        }

        for(int i = 0; i<2;i++){
            for(int j = 0; j<2;j++){
                for(int k = 0;k<15;k++){
                    for(int m = 0;m<7;m++){
                        for (int l = 0;l<10;l++){
                            if(vrstva2[i*15+k][m]<0.0){
                                kernely_1Dreal_2[i][l]-= 0.01 * alfa * vrstva1[k][l+m] * delta_2_3[j][m];
                            }else{
                                kernely_1Dreal_2[i][l]-= alfa * vrstva1[k][l+m] * delta_2_3[j][m];
                            }
                        }
                        if(vrstva2[i*15+k][m]<0.0){
                            bias_1Dreal_2[i] -= 0.01 * alfa * delta_2_3[j][m];
                        }else{
                            bias_1Dreal_2[i] -= alfa * delta_2_3[j][m];
                        }
                    }
                }
            }
        }

        for(int i = 0; i<5;i++){
            for(int j = 0; j<4;j++){
                    for(int m = 0;m<16;m++){
                        for (int l = 0;l<15;l++){
                            if(vrstva1[i][m]<0.0){
                                kernely_1Dreal_1[i][l]-= 0.01 * alfa * Q_kal_vstup[kroky +l+m] * delta_1_2[j][m];
                            }else{
                                kernely_1Dreal_1[i][l]-= alfa * Q_kal_vstup[kroky +l+m] * delta_1_2[j][m];
                            }
                        }
                        if(vrstva1[i][m]<0.0){
                            bias_1Dreal_1[i] -= 0.01 * alfa * delta_1_2[j][m];
                        }else{
                            bias_1Dreal_1[i] -= alfa * delta_1_2[j][m];
                        }
                    }
            
            }
        }

        for(int i = 0; i<5;i++){
            for(int j = 0; j<4;j++){
                    for(int m = 0;m<16;m++){
                        for (int l = 0;l<15;l++){
                            if(vrstva1[5+i][m]<0.0){
                                kernely_1Dreal_1[i][l]-= 0.01 * alfa * R_kal_vstup[kroky +l+m] * delta_1_2[j][m];
                            }else{
                                kernely_1Dreal_1[i][l]-= alfa * R_kal_vstup[kroky +l+m] * delta_1_2[j][m];
                            }
                        }
                        if(vrstva1[i][m]<0.0){
                            bias_1Dreal_1[i] -= 0.01 * alfa * delta_1_2[j][m];
                        }else{
                            bias_1Dreal_1[i] -= alfa * delta_1_2[j][m];
                        }
                    }
            
            }
        }

        for(int i = 0; i<5;i++){
            for(int j = 0; j<4;j++){
                    for(int m = 0;m<16;m++){
                        for (int l = 0;l<15;l++){
                            if(vrstva1[10+i][m]<0.0){
                                kernely_1Dreal_1[i][l]-= 0.01 * alfa * T_kal_vstup[kroky +l+m] * delta_1_2[j][m];
                            }else{
                                kernely_1Dreal_1[i][l]-= alfa * T_kal_vstup[kroky +l+m] * delta_1_2[j][m];
                            }
                        }
                        if(vrstva1[i][m]<0.0){
                            bias_1Dreal_1[i] -= 0.01 * alfa * delta_1_2[j][m];
                        }else{
                            bias_1Dreal_1[i] -= alfa * delta_1_2[j][m];
                        }
                    }
            
            }
        }


                
            }
        }
        }
    }
}
void NN::cnn1Dreal_val(int velic){
    vystupy.clear();
    std::vector<double> vystzkonv;
    std::vector<std::vector<double>>vrstva1;
    std::vector<std::vector<double>>vrstva2; 
    if(velic == 2){
        if (Q_val_vstup.size() != R_val_vstup.size()) {
            std::cout << "vstupni řady nejsou stejně dlouhý";
            exit(0);
        }
    
            for(int kroky = 0; kroky < (Q_val_vstup.size() - 30); kroky++){
                vystzkonv.clear();
                vrstva1.clear();
                vrstva2.clear();

                std::vector<double>mezivrst;
                for(int i = 0; i<5;i++){
                    for(int j = 0; j<16;j++){
                        double konvo = 0.0;
                        for(int k = 0; k<15;k++){
                            konvo += kernely_1Dreal_1[i][k] * Q_val_vstup[kroky +j+k];
                        }
                        konvo+=bias_1Dreal_1[i];
                        if(konvo<0.0){
                            mezivrst.push_back(konvo * 0.01);
                        }else{
                            mezivrst.push_back(konvo * 0.01);
                        }
                    }
                    vrstva1.push_back(mezivrst);
                    mezivrst.clear();
                }

                for(int i = 0; i<5;i++){
                    for(int j = 0; j<16;j++){
                        double konvo = 0.0;
                        for(int k = 0; k<15;k++){
                            konvo += kernely_1Dreal_1[i][k] * R_val_vstup[kroky +j+k];
                        }
                        konvo+=bias_1Dreal_1[i];
                        if(konvo<0.0){
                            mezivrst.push_back(konvo * 0.01);
                        }else{
                            mezivrst.push_back(konvo);
                        }
                    }
                    vrstva1.push_back(mezivrst);
                    mezivrst.clear();
                }
                

                for(int i = 0; i<2;i++){
                    for(int j = 0; j<10; j++){
                        for(int k = 0;k<7;k++){
                            double konvo = 0.0;
                            for(int l = 0; l<10;l++){
                                konvo+=kernely_1Dreal_2[i][l]*vrstva1[j][k+l];
                            }
                            konvo+=bias_1Dreal_2[i];
                            if(konvo<0.0){
                                mezivrst.push_back(konvo * 0.01);
                            }else{
                                mezivrst.push_back(konvo);
                            }
                        }
                        vrstva2.push_back(mezivrst);
                        mezivrst.clear();
                    }
                }

                for(int i = 0; i<2;i++){
                    for(int j = 0; j<20; j++){
                            double konvo = 0.0;
                            for(int l = 0; l<7;l++){
                                konvo+=kernely_1Dreal_3[i][l]*vrstva2[j][l];
                            }
                            konvo+=bias_1Dreal_3[i];
                            if(konvo<0.0){
                                vystzkonv.push_back(konvo * 0.01);
                            }else{
                                vystzkonv.push_back(konvo);
                            }
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
    }else if (velic == 3){
        if (Q_val_vstup.size() != R_val_vstup.size() || Q_val_vstup.size() != T_val_vstup.size()) {
            std::cout << "vstupni řady nejsou stejně dlouhý";
            exit(0);
        }
            for(int kroky = 0; kroky < (Q_val_vstup.size() - 30); kroky++){
                vystzkonv.clear();
                vrstva1.clear();
                vrstva2.clear();

                std::vector<double>mezivrst;
                for(int i = 0; i<5;i++){
                    for(int j = 0; j<16;j++){
                        double konvo = 0.0;
                        for(int k = 0; k<15;k++){
                            konvo += kernely_1Dreal_1[i][k] * Q_val_vstup[kroky +j+k];
                        }
                        konvo+=bias_1Dreal_1[i];
                        if(konvo<0.0){
                            mezivrst.push_back(konvo * 0.01);
                        }else{
                            mezivrst.push_back(konvo);
                        }
                    }
                    vrstva1.push_back(mezivrst);
                    mezivrst.clear();
                }

                for(int i = 0; i<5;i++){
                    for(int j = 0; j<16;j++){
                        double konvo = 0.0;
                        for(int k = 0; k<15;k++){
                            konvo += kernely_1Dreal_1[i][k] * R_val_vstup[kroky +j+k];
                        }
                        konvo+=bias_1Dreal_1[i];
                        if(konvo<0.0){
                            mezivrst.push_back(konvo * 0.01);
                        }else{
                            mezivrst.push_back(konvo);
                        }
                    }
                    vrstva1.push_back(mezivrst);
                    mezivrst.clear();
                }

                for(int i = 0; i<5;i++){
                    for(int j = 0; j<16;j++){
                        double konvo = 0.0;
                        for(int k = 0; k<15;k++){
                            konvo += kernely_1Dreal_1[i][k] * T_val_vstup[kroky +j+k];
                        }
                        konvo+=bias_1Dreal_1[i];
                        if(konvo<0.0){
                            mezivrst.push_back(konvo * 0.01);
                        }else{
                            mezivrst.push_back(konvo);
                        }
                    }
                    vrstva1.push_back(mezivrst);
                    mezivrst.clear();
                }

                for(int i = 0; i<2;i++){
                    for(int j = 0; j<15; j++){
                        for(int k = 0;k<7;k++){
                            double konvo = 0.0;
                            for(int l = 0; l<10;l++){
                                konvo+=kernely_1Dreal_2[i][l]*vrstva1[j][k+l];
                            }
                            konvo+=bias_1Dreal_2[i];
                            if(konvo<0.0){
                                mezivrst.push_back(konvo * 0.01);
                            }else{
                                mezivrst.push_back(konvo);
                            }
                        }
                        vrstva2.push_back(mezivrst);
                        mezivrst.clear();
                    }
                }

                for(int i = 0; i<2;i++){
                    for(int j = 0; j<30; j++){
                            double konvo = 0.0;
                            for(int l = 0; l<7;l++){
                                konvo+=kernely_1Dreal_3[i][l]*vrstva2[j][l];
                            }
                            konvo+=bias_1Dreal_3[i];
                            if(konvo<0.0){
                                vystzkonv.push_back(konvo * 0.01);
                            }else{
                                vystzkonv.push_back(konvo);
                            }
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
    }

double NN::random(double min, double max){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dis(min,max);
    return dis(gen);
}