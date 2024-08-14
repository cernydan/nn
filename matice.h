#ifndef MATICE_H
#define MATICE_H

#include <vector>
#include <stdexcept>
#include <iostream>

class Matice {
public:
    double** dta;
    size_t rows;
    size_t cols;

    Matice();    // Konstruktor bez rozměrů
    Matice(size_t rows, size_t cols);    // Konstruktor s rozměry
    ~Matice();    // Destruktor
    Matice(const Matice& other);     // Kopírovací konstruktor
    Matice(Matice&& other) noexcept;     // Move konstruktor
    Matice& operator=(const Matice& other);    // Kopírovací přiřazovací operátor
    Matice& operator=(Matice&& other) noexcept;    // Move přiřazovací operátor

    size_t getRows();
    size_t getCols();
    double getElement(size_t row, size_t col);

    void setElement(size_t row, size_t col, double value);
    void load_stdvv(const std::vector<std::vector<double>>& input);    // Načtení dat z std::vector<std::vector<double>>
    void printMat();

    double& operator()(size_t row, size_t col);
};

#endif // MATICE_H

