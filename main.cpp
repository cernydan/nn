#include "neural_net.h"
#include "neuron.h"
#include "neural_net.cpp"
#include "neuron.cpp"
#include "matice.h"
#include "matice.cpp"
#include <valarray>
#include <chrono>
#include <iostream>
using namespace std;
using namespace std::chrono;

int main(){

auto start = high_resolution_clock::now();

    NN mlp;
    mlp.set_rozmery({4,4,2,1});
    mlp.set_train_data({{0.1,0.2,0.3},{0.2,0.3,0.4},{0.3,0.4,0.5},{0.4,0.5,0.6},{0.5,0.6,0.7},{0.6,0.7,0.8}});
    mlp.chtenejout = {0.4,0.5,0.6,0.7,0.8,0.9};
    mlp.print_data();
    mlp.init_sit();
    mlp.online_bp(100000);
    for(int i= 0;i<mlp.vystupy.size();++i){
       std::cout<<mlp.vystupy[i];
    }
    mlp.count_cost();
    std::cout<<"\n\n"<<mlp.cost;


    //hhhh
    
    // Matice q;
    // q.load_stdvv({{0.1,0.2,0.3},{0.2,0.3,0.4},{0.3,0.4,0.5},{0.4,0.5,0.6},{0.5,0.6,0.7}});
    // q.printMat();
    // std::cout<<q.getElement(2,4);
    // std::cout<<"\n"<<q.dta[2];

auto stop = high_resolution_clock::now();

auto duration = duration_cast<microseconds>(stop - start);
std::cout <<"\n"<< duration.count() << endl;
};