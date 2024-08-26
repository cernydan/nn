#include "neural_net.h"
#include "neuron.h"
#include "neural_net.cpp"
#include "neuron.cpp"
#include "matice.h"
#include "tenzor.h"
#include <valarray>
#include <chrono>
#include <iostream>
using namespace std;
using namespace std::chrono;

int main(){

auto start = high_resolution_clock::now();

NN mlp;
mlp.set_vstup_rada({1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0});
mlp.udelej_radky(3,true);
mlp.dataprocnn_t.printTenzor();
mlp.kernely_t.printTenzor();
mlp.udelej_api(2,0.95,NN::lag,3,false);
mlp.dataprocnn_v2d[0].printMat();
mlp.kernely_v2d[0].printMat();
mlp.max_pool(mlp.dataprocnn_v2d[0],2,2).printMat();
mlp.avg_pool(mlp.dataprocnn_v2d[0],2,2).printMat();
mlp.konvo(mlp.dataprocnn_v2d[0],mlp.kernely_v2d[0]).printMat();

auto stop = high_resolution_clock::now();

auto duration = duration_cast<microseconds>(stop - start);
std::cout <<"\n"<< duration.count() << endl;
};