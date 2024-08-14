#include "matice.h"
#include <iostream>
#include <algorithm>

using namespace std;

Matice::Matice() :     // Konstruktor bez rozměrů
    dta(nullptr), 
    rows(0), 
    cols(0) {}

Matice::Matice(size_t rows, size_t cols) :     // Konstruktor s rozměry
    rows(rows), 
    cols(cols) {
    dta = new double*[rows];
    for (size_t i = 0; i < rows; ++i) {
        dta[i] = new double[cols]();
    }
}

Matice::~Matice() {        // Destruktor

    delete[] dta;
}

Matice::Matice(const Matice& other) :         // Kopírovací konstruktor
    rows(other.rows), 
    cols(other.cols) {
    dta = new double*[rows];
    for (size_t i = 0; i < rows; ++i) {
        dta[i] = new double[cols];
        std::copy(other.dta[i], other.dta[i] + cols, dta[i]);
    }
}

Matice::Matice(Matice&& other) noexcept :     // Move konstruktor
    dta(other.dta), 
    rows(other.rows), 
    cols(other.cols) {
    other.dta = nullptr;
    other.rows = 0;
    other.cols = 0;
}

Matice& Matice::operator=(const Matice& other) {    // Kopírovací přiřazovací operátor
    if (this != &other) {
        delete[] dta;

        rows = other.rows;
        cols = other.cols;
        dta = new double*[rows];
        for (size_t i = 0; i < rows; ++i) {
            dta[i] = new double[cols];
            std::copy(other.dta[i], other.dta[i] + cols, dta[i]);
        }
    }
    return *this;
}

Matice& Matice::operator=(Matice&& other) noexcept {    // Move přiřazovací operátor
    if (this != &other) {
        delete[] dta;

        dta = other.dta;
        rows = other.rows;
        cols = other.cols;

        other.dta = nullptr;
        other.rows = 0;
        other.cols = 0;
    }
    return *this;
}

void Matice::load_stdvv(const std::vector<std::vector<double>>& input) {
    if (!input.empty()) {
        rows = input.size();
        cols = input[0].size();
        delete[] dta;

        dta = new double*[rows];
        for (size_t i = 0; i < rows; ++i) {
            dta[i] = new double[cols];
            if (input[i].size() != cols) {
                std::cout<<"rozmery vstupu neodpovidaji matici";
                exit(0);
            }
            std::copy(input[i].begin(), input[i].end(), dta[i]);
        }
    }
}

size_t Matice::getRows() {
    return rows;
}

size_t Matice::getCols() {
    return cols;
}

double Matice::getElement(size_t row, size_t col) {
    if (row >= rows || col >= cols) {
        std::cout<<"index mimo rozmery matice";
        exit(0);
    }
    return dta[row][col];
}

void Matice::setElement(size_t row, size_t col, double value) {
    if (row >= rows || col >= cols) {
        std::cout<<"index mimo rozmery matice";
        exit(0);
    }
    dta[row][col] = value;
}

void Matice::printMat() {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            std::cout << getElement(i, j) << " ";
        }
        std::cout << std::endl;
    }
}

double& Matice::operator()(size_t row, size_t col) {
    if (row >= rows || col >= cols) {
        std::cout<<"index mimo rozmery matice";
        exit(0);
    }
    return dta[row][col];
}


