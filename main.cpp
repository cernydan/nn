#include "neural_net.h"
#include "neuron.h"
#include "neural_net.cpp"
#include "neuron.cpp"
#include "lstmneuron.h"
#include "lstmneuron.cpp"
#include "matice.h"
#include "tenzor.h"
#include "threadpool.h"
#include "threadpool.cpp"
#include <valarray>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include <mutex>
using namespace std;
using namespace std::chrono;

template<typename T>
void nastav(long int kolik,Tenzor<T>& tenzor,const Matice<T>& matice){
    if(kolik>tenzor.getDepth()){
        std::cout << "Moc" << std::endl;
        exit(0);
    }

    for(int i = 0; i<kolik;i++){
        tenzor.set_matrix(i,matice);
    }
}

template<typename T>
void nastav_t(long int start, long int konec,Tenzor<T>& tenzor,const Matice<T>& matice){
    if(konec>tenzor.getDepth()){
        std::cout << "Moc" << std::endl;
        exit(0);
    }


    for(int i = start; i<konec;i++){
        tenzor.set_matrix(i,matice);
    }
}

int main() {

    auto start = high_resolution_clock::now();



///////////////LSTM ///////////////////////////////////////////
    // NN abcde;
    // abcde.set_train_data({{0.1,0.2,0.3,0.4,0.5},{0.2,0.3,0.4,0.5,0.6},{0.3,0.4,0.5,0.6,0.7},{0.4,0.5,0.6,0.7,0.8},{0.5,0.6,0.7,0.8,0.9}});
    // abcde.set_chtenejout({0.6,0.7,0.8,0.9,1});
    // abcde.lstm_1cell(1,100);
    // abcde.print_vystup();





//    Matice<double> q(90,90);
//     q.rand_vypln(0,1);
//    Tenzor<double> z(120000,90,90);
//   //////////////////////////////////////////////OBYČ SERIE
//    //nastav(120000,z,q);

//    ////////////////////////////////////////////THREAD POOL

//     int jadra = std::thread::hardware_concurrency();
//     int kus = 120000 / jadra;

//     // Vytvoření ThreadPoolu s počtem jader
//     ThreadPool pool(jadra);

//     // Vector pro uchování future objektů
//     std::vector<std::future<void>> futures;

//     // Přidání úloh do poolu
//     for (int i = 0; i < jadra; ++i) {
//         futures.push_back(pool.enqueueTask([i, kus, &z, &q]() {
//             nastav_t<double>(i * kus, (i + 1) * kus, z, q);
//         }));
//     }

//     // Čekání na dokončení všech úloh
//     for (auto& future : futures) {
//         future.get();
//     }


////////////////////////// JENOM THREAD

    // int jadra = std::thread::hardware_concurrency();
    // int kus = 120000/jadra;
    // std::vector<std::thread> threads;

    // for(int j = 0;j<jadra;j++){
    //     threads.push_back(std::thread(nastav_t<double>, j * kus, (j + 1) * kus, std::ref(z), std::ref(q)));
    // }

    // for (auto& t : threads) {
    //     t.join();
    // }

    //std::cout<<z.getElement(90000,40,40);


auto stop = high_resolution_clock::now();

auto duration = duration_cast<microseconds>(stop - start);
std::cout <<"\n"<< duration.count()/1000000 << endl;

};