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

Matice<double> NN::max_pool(Matice<double> vstupnim, size_t oknorad, size_t oknosl){
    size_t radky = vstupnim.getRows()-oknorad+1;
    size_t sloupce = vstupnim.getCols()-oknosl+1;
    Matice<double>vystupm(radky,sloupce);
    double tedmax;
    for(int i = 0;i<radky;i++){
        for(int j = 0;j<sloupce;j++){
            for(int k = 0;k<oknorad;k++){
                for(int l = 0;l<oknosl;l++){
                    if((k)==0 and (l) == 0){
                        tedmax = vstupnim.getElement(i,j);
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

Matice<double> NN::max_pool_fullstep(Matice<double> vstupnim, size_t oknorad, size_t oknosl){
    size_t radky = vstupnim.getRows()/oknorad;
    size_t sloupce = vstupnim.getCols()/oknosl;
    Matice<double>vystupm(radky,sloupce);
    double tedmax;
    for(int i = 0;i<radky;i++){
        for(int j = 0;j<sloupce;j++){
            for(int k = 0;k<oknorad;k++){
                for(int l = 0;l<oknosl;l++){
                    if(k==0 and l == 0){
                        tedmax = vstupnim.getElement(i*oknorad,j*oknosl);
                    }else{
                        if(tedmax<vstupnim.getElement(i*oknorad+k,j*oknosl+l)){
                            tedmax = vstupnim.getElement(i*oknorad+k,j*oknosl+l);
                        }
                    }
                }
            }
            vystupm.setElement(i,j,tedmax);
        }
    }
    return vystupm;
}

Tenzor<double> NN::max_pool_fullstep_3d(Tenzor<double> vstupnim, size_t oknorad, size_t oknosl){
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

void NN::online_bp_th(int iter) {
    int jadra = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    for (int m = 0; m < iter; ++m) {
        std::cout << m << "\n";
        vystupy.clear();
        pom_vystup.clear();

        for (int l = 0; l < train_data.size(); ++l) {
            int kusneur = rozmery[0]/jadra;

            // Paralelizace výpočtů neuronů první vrstvy
            for(int i = 0;i<jadra;++i){
                int start = i*kusneur;
                int end = (i == jadra - 1) ? rozmery[0] : start + kusneur;
                threads.push_back(std::thread([this, l, i,start,end](){
                    for(int j = start;j<end;++j){
                        sit[0][j].set_vstupy(train_data[l]);
                        sit[0][j].vypocet();
                    }
                }));
            }

            for (auto& t : threads) {
              t.join();
            }

            threads.clear();

            // Uložení výsledků první vrstvy
            for (int i = 0; i < rozmery[0]; ++i) {
                pom_vystup.push_back(sit[0][i].o);
            }

            // Paralelizace výpočtů v dalších vrstvách
            for (int i = 1; i < pocet_vrstev; ++i) {
                kusneur = rozmery[i]/jadra;

                for(int j = 0;j<jadra;++j){
                    int start = j*kusneur;
                    int end = (j == jadra - 1) ? rozmery[i] : start + kusneur;
                    threads.push_back(std::thread([this, l, i,j,start,end]() {
                        for(int k = start;k<end;++k){
                            sit[i][k].set_vstupy(pom_vystup);
                            sit[i][k].vypocet();
                        }
                }));
            }


               for (auto& t : threads) {
              t.join();
            }
                pom_vystup.clear();
                threads.clear();

                // Uložení výstupů aktuální vrstvy
                for (int j = 0; j < rozmery[i]; ++j) {
                    pom_vystup.push_back(sit[i][j].o);
                }
            }
            vystupy.push_back(pom_vystup[0]);
                            // dava smysl jen pro 1 vystup

            // BP - Backpropagation výpočty delta hodnot
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
    }
}

void NN::lstm_1cell(int batch_size,int iter){
    LSTMNeuron pepa;
    pepa.set_vstupy(train_data[0]);
    pepa.set_randomvahy();
    for(int m = 0;m<iter;++m){
        int pozice = 0;
        vystupy.clear();
        for(int k = 0;k<(train_data.size()/batch_size);++k){
            for (int i = pozice;i<(pozice+batch_size);++i ){
                pepa.set_vstupy(train_data[i]);
                pepa.vypocet();
                vystupy.push_back(pepa.vystup);
            }

            for(int i =(batch_size-1);i>=0;--i){
                pepa.dLdz = pepa.vystup_hist[i] - chtenejout[i+pozice];
                pepa.dLda = pepa.Wy*pepa.dLdz + pepa.da;
                pepa.dLdc = pepa.dc + pepa.dLda*pepa.output_hist[i]*(1-pow(tanh(pepa.longterm_hist[i]),2));
                pepa.dLdcan = pepa.dLdc*pepa.update_hist[i]*(1-pow(pepa.candidate_hist[i],2));
                pepa.dLdu = pepa.dLdc*pepa.candidate_hist[i]*(pepa.update_hist[i]*(1-pepa.update_hist[i]));
                pepa.dLdf = pepa.dLdc*pepa.candidate_hist[i]*(pepa.forget_hist[i]*(1-pepa.forget_hist[i]));
                pepa.dLdo = pepa.dLda*tanh(pepa.longterm_hist[i])*(pepa.output_hist[i]*(1-pepa.output_hist[i]));
                pepa.dc = (pepa.dc+pepa.dLda*pepa.output_hist[i]*(1-pow(tanh(pepa.longterm_hist[i]),2)))*pepa.forget_hist[i];
                pepa.da = pepa.candidate.vahy[pepa.candidate.vahy.size()]*pepa.dLdcan +
                        pepa.update.vahy[pepa.update.vahy.size()]*pepa.dLdu + 
                        pepa.forget.vahy[pepa.forget.vahy.size()]*pepa.dLdf + 
                        pepa.output.vahy[pepa.output.vahy.size()]*pepa.dLdo;
                for(int j = 0;j<(pepa.forget.vahy.size()-2);++j){
                    pepa.dLdx.push_back(
                        pepa.candidate.vahy[j]*pepa.dLdcan +
                        pepa.update.vahy[j]*pepa.dLdu + 
                        pepa.forget.vahy[j]*pepa.dLdf + 
                        pepa.output.vahy[j]*pepa.dLdo
                    );

                    pepa.candidate.vahy[j] = pepa.candidate.vahy[j] - alfa*pepa.dLdcan*train_data[i+pozice][j];
                    pepa.update.vahy[j] = pepa.update.vahy[j] - alfa*pepa.dLdu*train_data[i+pozice][j]; 
                    pepa.forget.vahy[j] = pepa.forget.vahy[j] - alfa*pepa.dLdf*train_data[i+pozice][j];
                    pepa.output.vahy[j] = pepa.output.vahy[j] - alfa*pepa.dLdo*train_data[i+pozice][j];

                }
                pepa.candidate.vahy[pepa.candidate.vahy.size()-2] = pepa.candidate.vahy[pepa.candidate.vahy.size()-2]-alfa*pepa.dLdcan;
                pepa.update.vahy[pepa.update.vahy.size()-2] = pepa.update.vahy[pepa.update.vahy.size()-2] - alfa*pepa.dLdu; 
                pepa.forget.vahy[pepa.forget.vahy.size()-2] = pepa.forget.vahy[pepa.forget.vahy.size()-2] - alfa*pepa.dLdf;
                pepa.output.vahy[pepa.output.vahy.size()-2] = pepa.output.vahy[pepa.output.vahy.size()-2] - alfa*pepa.dLdo;
                pepa.by = pepa.by - alfa* pepa.dLdz;
                pepa.Wy = pepa.Wy - alfa* pepa.dLdz* pepa.shortterm_hist[i];

                if(i != 0){
                    pepa.candidate.vahy[pepa.candidate.vahy.size()-1] = pepa.candidate.vahy[pepa.candidate.vahy.size()-1]-
                                                                        alfa*pepa.dLdcan*pepa.shortterm_hist[i-1];
                    pepa.update.vahy[pepa.update.vahy.size()-1] = pepa.update.vahy[pepa.update.vahy.size()-1] - 
                                                                alfa*pepa.dLdu*pepa.shortterm_hist[i-1]; 
                    pepa.forget.vahy[pepa.forget.vahy.size()-1] = pepa.forget.vahy[pepa.forget.vahy.size()-1] - 
                                                                alfa*pepa.dLdf*pepa.shortterm_hist[i-1];
                    pepa.output.vahy[pepa.output.vahy.size()-1] = pepa.output.vahy[pepa.output.vahy.size()-1] - 
                                                                alfa*pepa.dLdo*pepa.shortterm_hist[i-1];
                }
            }
            pozice += batch_size;
            pepa.forget_hist.clear();
            pepa.update_hist.clear();
            pepa.candidate_hist.clear();
            pepa.output_hist.clear();
            pepa.shortterm_hist.clear();
            pepa.longterm_hist.clear();
            pepa.vystup_hist.clear();
            pepa.dLdx.clear();
        }
    }
}

Matice<double> NN::udelej_api(int n, double beta, Co coze, int kolik){
    std::vector<double> api(vstupni_cr.size()-n+1);
    Matice<double> nova;
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
            nova.resize(nrows,ncols);
            for (int i = 0; i<nrows;++i){
                for (int j = 0; j<ncols;++j){
                    nova.setElement(i,j,api[por]);
                    por+=1;
                }
             }
            break;
        }
        case lag:{
            size_t nrows  = api.size() - kolik+1;
            size_t ncols = kolik;
            nova.resize(nrows,ncols);
            for (int i = 0; i<nrows;++i){
                for (int j = 0; j<ncols;++j){
                    nova.setElement(i,j,api[i+j]);
                }
            }
        break;
        }
    }
return nova;
}

Matice<double> NN::udelej_prumery(int n, Co coze, int kolik){
    std::vector<double> prumery(vstupni_cr.size()-(2*n));
    Matice<double> nova;
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
            nova.resize(nrows,ncols);
            for (int i = 0; i<nrows;++i){
                for (int j = 0; j<ncols;++j){
                    nova.setElement(i,j,prumery[por]);
                    por+=1;
                }
             }
        break;
        }
        case lag:{
            size_t nrows  = prumery.size() - kolik+1;
            size_t ncols = kolik;
            nova.resize(nrows,ncols);
            for (int i = 0; i<nrows;++i){
                for (int j = 0; j<ncols;++j){
                    nova.setElement(i,j,prumery[i+j]);
                }
            }
        break;
        }
    }
    return nova;
}

void NN::online_bp_thread(int iter) {
    int jadra = std::thread::hardware_concurrency();
    ThreadPool pool(jadra);
    for (int m = 0; m < iter; ++m) {
        std::cout << m << "\n";
        vystupy.clear();
        pom_vystup.clear();

        for (int l = 0; l < train_data.size(); ++l) {
            std::vector<std::future<void>> futures;
            int kusneur = rozmery[0]/jadra;

            // Paralelizace výpočtů neuronů první vrstvy
            for(int i = 0;i<jadra;++i){
                int start = i*kusneur;
                int end = (i == jadra - 1) ? rozmery[0] : start + kusneur;
                futures.push_back(pool.enqueueTask([this, l, i,start,end]() {
                    for(int j = start;j<end;++j){
                        sit[0][j].set_vstupy(train_data[l]);
                        sit[0][j].vypocet();
                    }
                }));
            }

            // Čekání na dokončení všech výpočtů pro první vrstvu
            for (auto& future : futures) {
                future.get();
            }
            futures.clear();

            // Uložení výsledků první vrstvy
            for (int i = 0; i < rozmery[0]; ++i) {
                pom_vystup.push_back(sit[0][i].o);
            }

            // Paralelizace výpočtů v dalších vrstvách
            for (int i = 1; i < pocet_vrstev; ++i) {
                kusneur = rozmery[i]/jadra;
                futures.clear();

                for(int j = 0;j<jadra;++j){
                    int start = j*kusneur;
                    int end = (j == jadra - 1) ? rozmery[i] : start + kusneur;
                    futures.push_back(pool.enqueueTask([this, l, i,j,start,end]() {
                        for(int k = start;k<end;++k){
                            sit[i][k].set_vstupy(pom_vystup);
                            sit[i][k].vypocet();
                        }
                }));
            }


                for (auto& future : futures) {
                    future.get();
                }
                pom_vystup.clear();

                // Uložení výstupů aktuální vrstvy
                for (int j = 0; j < rozmery[i]; ++j) {
                    pom_vystup.push_back(sit[i][j].o);
                }
            }
            vystupy.push_back(pom_vystup[0]);
                            // dava smysl jen pro 1 vystup

            // BP - Backpropagation výpočty delta hodnot
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
    }
}




//[[Rcpp::export]]
void nn_online_bp_th(Rcpp::XPtr<NN> nn,int iters){
    nn->online_bp_th(iters);
}

//[[Rcpp::export]]
void lstm_1cell(Rcpp::XPtr<NN> nn,int batch_size,int iters){
    nn->lstm_1cell(batch_size,iters);
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
