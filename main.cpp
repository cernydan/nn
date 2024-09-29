#include "neural_net.h"
#include "neuron.h"
#include "neural_net.cpp"
#include "neuron.cpp"
#include "matice.h"
#include "tenzor.h"
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

    

   Matice<double> q;
   q.load_stdvv({{1,2,3,4,5},{1,2,3,4,5},{1,2,3,4,5},{1,2,3,4,5},{1,2,3,4,5}});
   Tenzor<double> z(25000000,5,5);
   q.printMat();
    nastav(25000000,z,q);



    // int jadra = 5;
    // int kus = 25000000/jadra;
    // std::vector<std::thread> threads;

    // for(int j = 0;j<jadra;j++){
    //     threads.push_back(std::thread(nastav_t<double>, j * kus, (j + 1) * kus, std::ref(z), std::ref(q)));
    // }

    // for (auto& t : threads) {
    //     t.join();
    // }



auto stop = high_resolution_clock::now();

auto duration = duration_cast<microseconds>(stop - start);
std::cout <<"\n"<< duration.count()/1000000 << endl;

};