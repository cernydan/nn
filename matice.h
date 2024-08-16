#ifndef MATICE_H
#define MATICE_H

#include <vector>
#include <stdexcept>
#include <iostream>

template<typename T>  // šablona pro třídu s libovolným datovým typem - dosazení za T
class Matice {
public:
    T** dta;
    size_t rows;
    size_t cols;

    Matice();    // Konstruktor bez rozměrů
    Matice(size_t rows, size_t cols);    // Konstruktor s rozměry
    ~Matice();    // Destruktor
    Matice(const Matice& other);     // Kopírovací konstruktor
    Matice(Matice&& other) noexcept;     // Move konstruktor
    Matice& operator=(const Matice& other);    // Kopírovací přiřazovací operátor
    Matice& operator=(Matice&& other) noexcept;    // Move přiřazovací operátor

        // Accessors - přístup k prvkům indexy
    T& operator()(size_t row, size_t col);           // & adresa prvku - umožňuje změnu/zápis
    T operator()(size_t row, size_t col) const;     // const - jen čtení
    
    size_t getRows();
    size_t getCols();
    T getElement(size_t row, size_t col);

    void setElement(size_t row, size_t col, T value);
    void load_stdvv(const std::vector<std::vector<T>>& input);    // Načtení dat z std::vector<std::vector<double>>
    void printMat();

    
};

#include "matice.tpp" // U templatu include tady

#endif // MATICE_H

