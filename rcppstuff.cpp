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
void nn_online_bp_th(Rcpp::XPtr<NN> nn,int iters){
    nn->online_bp_th(iters);
}

//[[Rcpp::export]]
void lstm_1cell(Rcpp::XPtr<NN> nn,int batch_size,int iters){
    nn->lstm_1cell(batch_size,iters);
}

//[[Rcpp::export]]
void nn_online_bp_adam(Rcpp::XPtr<NN> nn,int iters){
    nn->online_bp_adam(iters);
}

//[[Rcpp::export]]
void nn_cnn_pokus_cal(Rcpp::XPtr<NN> nn,size_t vel_ker, size_t poc_ker,int iters){
    nn->cnnonfly_cal(vel_ker, poc_ker, iters);
}

//[[Rcpp::export]]
void nn_cnn_pokus_val(Rcpp::XPtr<NN> nn){
    nn->cnnonfly_val();
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
void nn_set_vstup_rada(Rcpp::XPtr<NN> nn, Rcpp::NumericVector input_cr){
    std::vector<double> input_cr_c = Rcpp::as<std::vector<double>>(input_cr);
    nn->set_vstup_rada(input_cr_c);
}

// [[Rcpp::export]]
void nn_udelej_radky(Rcpp::XPtr<NN> nn, size_t velrad){
    nn->udelej_radky(velrad);
}

// [[Rcpp::export]]
void nn_udelej_lag(Rcpp::XPtr<NN> nn, size_t lag){
    nn->udelej_lag(lag);
}

// [[Rcpp::export]]
void nn_udelej_api(Rcpp::XPtr<NN> nn,int n, double beta, int coze, int kolik) {

  NN::Co coze_enum;
  if (coze == 0) {
    coze_enum = NN::radky;
  } else if (coze == 1) {
    coze_enum = NN::lag;
  } else {
    Rcpp::stop("Neplatná hodnota pro 'coze'");
  }
  
  nn->udelej_api(n, beta, coze_enum, kolik);
}

// [[Rcpp::export]]
void nn_udelej_prumery(Rcpp::XPtr<NN> nn, int n, int coze, int kolik) {

  NN::Co coze_enum;
  if (coze == 0) {
    coze_enum = NN::radky;
  } else if (coze == 1) {
    coze_enum = NN::lag;
  } else {
    Rcpp::stop("Neplatná hodnota pro 'coze'");
  }
  
  nn->udelej_prumery(n, coze_enum, kolik);
}


//[[Rcpp::export]]
void nn_shuffle_train(Rcpp::XPtr<NN> nn){
    nn->shuffle_train();
}
