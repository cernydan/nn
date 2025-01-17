#pragma once
#ifndef TENZOR_H
#define TENZOR_H

#include <vector>
#include <stdexcept>
#include <iostream>
#include "matice.h"

template<typename T>  // šablona pro třídu s libovolným datovým typem - dosazení za T
class Tenzor {
public:
    T*** dta;
    size_t depth,cols,rows;

    Tenzor();    // Konstruktor bez rozměrů
    Tenzor(size_t depth, size_t rows, size_t cols);    // Konstruktor s rozměry
    ~Tenzor();    // Destruktor
    Tenzor(const Tenzor& other);     // Kopírovací konstruktor
    Tenzor(Tenzor&& other) noexcept;     // Move konstruktor
    Tenzor& operator=(const Tenzor& other);    // Kopírovací přiřazovací operátor
    Tenzor& operator=(Tenzor&& other) noexcept;    // Move přiřazovací operátor

        // Accessors - přístup k prvkům indexy
    T& operator()(size_t depth, size_t row, size_t col);           // & adresa prvku - umožňuje změnu/zápis
    T operator()(size_t depth, size_t row, size_t col) const;     // const - jen čtení
    
    size_t getDepth() const;
    size_t getRows()  const;
    size_t getCols() const;
    T getElement(size_t depth, size_t row, size_t col);

    void setElement(size_t depth, size_t row, size_t col, T value);
    void load_stdvv(const std::vector<std::vector<std::vector<T>>>& input);    // Načtení dat z std::v<std::v<std::v>>
    void printTenzor();
    void printRozmery();
    void add_matrix(const Matice<T>& matice);
    void set_matrix(size_t depth, const Matice<T>& matice);
    void rand_vypln(double min, double max);
    void flip180();
    void obal_nul(size_t layers);
    void dilace(size_t row_pad, size_t col_pad);
    void resize(size_t hloubka, size_t radky, size_t sloupce);
    
};

#include "tenzor.tpp" // U templatu include tady

#endif // Tenzor_H