#include <Rcpp.h>
#include "neuron.h"
#include "neural_net.h"


// [[Rcpp::export]]
Rcpp::XPtr<Neuron> udelej_neuron() {
    return Rcpp::XPtr<Neuron>(new Neuron());
}


//[[Rcpp::export]]
void neuron_set_vstupy(Rcpp::XPtr<Neuron> neuron, Rcpp::NumericVector inputs){
    std::vector<double> inputs_c = Rcpp::as<std::vector<double>>(inputs);
    neuron->set_vstupy(inputs_c);
}


//[[Rcpp::export]]
void neuron_print_neuron(Rcpp::XPtr<Neuron> neuron){
    neuron->print_neuron();
}

//[[Rcpp::export]]
void neuron_set_randomvahy(Rcpp::XPtr<Neuron> neuron) {
    neuron->set_randomvahy();
}

//[[Rcpp::export]]
void neuron_set_rucovahy(Rcpp::XPtr<Neuron> neuron, Rcpp::NumericVector weights){
    std::vector<double> weights_c = Rcpp::as<std::vector<double>>(weights);
    if(weights_c.size() != neuron->vstupy.size()){
        Rcpp::stop("Pocet vah neodpovida poctu vstupu");
    }
    neuron->set_rucovahy(weights_c);
}

//[[Rcpp::export]]
void neuron_vypocet(Rcpp::XPtr<Neuron> neuron) {
    neuron->vypocet();
}

//[[Rcpp::export]]
double neuron_get_vystup(Rcpp::XPtr<Neuron> neuron) {
    
    return neuron->o;
}

//[[Rcpp::export]]
double neuron_get_aktiv(Rcpp::XPtr<Neuron> neuron) {
    
    return neuron->a;
}

// [[Rcpp::export]]
Rcpp::NumericVector neuron_get_vahy(Rcpp::XPtr<Neuron> neuron) {
    return Rcpp::wrap(neuron->vahy);
}

/////////////////////////    NEURAL NET     //////////////////////////////////////////////////////////

// [[Rcpp::export]]
Rcpp::XPtr<NN> udelej_nn() {
    return Rcpp::XPtr<NN>(new NN());
}


//[[Rcpp::export]]
void nn_set_chtenejout(Rcpp::XPtr<NN> nn, Rcpp::NumericVector obsout){
    std::vector<double> obsout_c = Rcpp::as<std::vector<double>>(obsout);
    nn->set_chtenejout(obsout_c);
}

//[[Rcpp::export]]
void nn_print_nn(Rcpp::XPtr<NN> nn){
    nn->print_nn();
}

//[[Rcpp::export]]
void nn_init_nn(Rcpp::XPtr<NN> nn,int pocet_vst, Rcpp::NumericVector rozmers){
    std::vector<int> rozmers_c = Rcpp::as<std::vector<int>>(rozmers);
    nn->init_sit(pocet_vst,rozmers_c);
}

//[[Rcpp::export]]
void nn_online_bp(Rcpp::XPtr<NN> nn,int iters){
    nn->online_bp(iters);
}


//[[Rcpp::export]]
void nn_online_bp_adam(Rcpp::XPtr<NN> nn,int iters){
    nn->online_bp_adam(iters);
}

//[[Rcpp::export]]
void nn_set_traindata(Rcpp::XPtr<NN> nn, Rcpp::NumericMatrix ddata) {
    std::vector<std::vector<double>> ddata_c(ddata.nrow(), std::vector<double>(ddata.ncol()));
    for (int i = 0; i < ddata.nrow(); ++i) {
        for (int j = 0; j < ddata.ncol(); ++j) {
            ddata_c[i][j] = ddata(i, j);
        }
    }
    nn->set_train_data(ddata_c);
}

//[[Rcpp::export]]
void nn_set_valdata(Rcpp::XPtr<NN> nn, Rcpp::NumericMatrix ddata) {
    std::vector<std::vector<double>> ddata_c(ddata.nrow(), std::vector<double>(ddata.ncol()));
    for (int i = 0; i < ddata.nrow(); ++i) {
        for (int j = 0; j < ddata.ncol(); ++j) {
            ddata_c[i][j] = ddata(i, j);
        }
    }
    nn->set_val_data(ddata_c);
}

//[[Rcpp::export]]
void nn_print_data(Rcpp::XPtr<NN> nn){
    nn->print_data();
}

//[[Rcpp::export]]
void nn_valid(Rcpp::XPtr<NN> nn){
    nn->valid();
}

// [[Rcpp::export]]
Rcpp::NumericVector nn_get_vystupy(Rcpp::XPtr<NN> nn) {
    return Rcpp::wrap(nn->vystupy);
}

// [[Rcpp::export]]
double nn_count_cost(Rcpp::XPtr<NN> nn){
    nn->count_cost();
    return nn->cost;
}

//[[Rcpp::export]]
void nn_set_vstup_rady(Rcpp::XPtr<NN> nn, Rcpp::NumericVector Qkal_r, Rcpp::NumericVector Qval_r,
                                        Rcpp::NumericVector Rkal_r, Rcpp::NumericVector Rval_r,
                                        Rcpp::NumericVector Tkal_r, Rcpp::NumericVector Tval_r){
    std::vector<double> Qkal_c = Rcpp::as<std::vector<double>>(Qkal_r);
    std::vector<double> Qval_c = Rcpp::as<std::vector<double>>(Qval_r);
    std::vector<double> Rkal_c = Rcpp::as<std::vector<double>>(Rkal_r);
    std::vector<double> Rval_c = Rcpp::as<std::vector<double>>(Rval_r);
    std::vector<double> Tkal_c = Rcpp::as<std::vector<double>>(Tkal_r);
    std::vector<double> Tval_c = Rcpp::as<std::vector<double>>(Tval_r);
    nn->set_vstup_rady(Qkal_c,Qval_c,Rkal_c,Rval_c,Tkal_c,Tval_c);
}

//[[Rcpp::export]]
void nn_shuffle_train(Rcpp::XPtr<NN> nn){
    nn->shuffle_train();
}

//[[Rcpp::export]]
void nn_cnn_onfly_cal(Rcpp::XPtr<NN> nn,size_t vel_ker, size_t poc_ker,int iters){
    nn->cnnonfly_cal(vel_ker, poc_ker, iters);
}

//[[Rcpp::export]]
void nn_cnn_onfly_val(Rcpp::XPtr<NN> nn){
    nn->cnnonfly_val();
}

//[[Rcpp::export]]
void nn_cnn_1d_cal(Rcpp::XPtr<NN> nn,size_t vel_ker, size_t poc_ker,int iters){
    nn->cnn1D_cal(vel_ker, poc_ker, iters);
}

//[[Rcpp::export]]
void nn_cnn_1d_val(Rcpp::XPtr<NN> nn){
    nn->cnn1D_val();
}

//[[Rcpp::export]]
void nn_cnn_full_cal(Rcpp::XPtr<NN> nn,int iters){
    nn->cnn_full_cal(iters);
}

//[[Rcpp::export]]
void nn_cnn_full_val(Rcpp::XPtr<NN> nn){
    nn->cnn_full_val();
}