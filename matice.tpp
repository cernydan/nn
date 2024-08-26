#include "matice.h"

template<typename T>
Matice<T>::Matice() :     // Konstruktor bez rozměrů
    dta(nullptr), 
    rows(0), 
    cols(0) {}

template<typename T>
Matice<T>::Matice(size_t rows, size_t cols) :     // Konstruktor s rozměry
    rows(rows), 
    cols(cols) {
    dta = new T*[rows];
    for (size_t i = 0; i < rows; ++i) {
        dta[i] = new T[cols]();
    }
}

template<typename T>
Matice<T>::~Matice() {        // Destruktor

    for (size_t i = 0; i < rows; ++i) {
        delete[] dta[i];
    }
    delete[] dta;
}

template<typename T>
Matice<T>::Matice(const Matice& other) :         // Kopírovací konstruktor
    rows(other.rows), 
    cols(other.cols) {
    dta = new T*[rows];
    for (size_t i = 0; i < rows; ++i) {
        dta[i] = new T[cols];
        std::copy(other.dta[i], other.dta[i] + cols, dta[i]);
    }
}

template<typename T>
Matice<T>::Matice(Matice&& other) noexcept :     // Move konstruktor
    dta(other.dta), 
    rows(other.rows), 
    cols(other.cols) {
    other.dta = nullptr;
    other.rows = 0;
    other.cols = 0;
}

template<typename T>
Matice<T>& Matice<T>::operator=(const Matice& other) {    // Kopírovací přiřazovací operátor
    if (this != &other) {
        for (size_t i = 0; i < rows; ++i) {
            delete[] dta[i];
        }
        delete[] dta;

        rows = other.rows;
        cols = other.cols;
        dta = new T*[rows];
        for (size_t i = 0; i < rows; ++i) {
            dta[i] = new T[cols];
            std::copy(other.dta[i], other.dta[i] + cols, dta[i]);
        }
    }
    return *this;
}

template<typename T>
Matice<T>& Matice<T>::operator=(Matice&& other) noexcept {    // Move přiřazovací operátor
    if (this != &other) {
        for (size_t i = 0; i < rows; ++i) {
            delete[] dta[i];
        }
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

template<typename T>
void Matice<T>::load_stdvv(const std::vector<std::vector<T>>& input) {
    if (!input.empty()) {
        rows = input.size();
        cols = input[0].size();
        delete[] dta;

        dta = new T*[rows];
        for (size_t i = 0; i < rows; ++i) {
            dta[i] = new T[cols];
            if (input[i].size() != cols) {
                std::cout<<"rozmery vstupu neodpovidaji matici";
                exit(0);
            }
            std::copy(input[i].begin(), input[i].end(), dta[i]);
        }
    }
}

template<typename T>
size_t Matice<T>::getRows() const {
    return rows;
}

template<typename T>
size_t Matice<T>::getCols() const {
    return cols;
}

template<typename T>
T Matice<T>::getElement(size_t row, size_t col) {
    if (row >= rows || col >= cols) {
        std::cout<<"index mimo rozmery matice";
        exit(0);
    }
    return dta[row][col];
}

template<typename T>
void Matice<T>::setElement(size_t row, size_t col, T value) {
    if (row >= rows || col >= cols) {
        std::cout<<"index mimo rozmery matice";
        exit(0);
    }
    dta[row][col] = value;
}

template<typename T>
void Matice<T>::printMat() {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            std::cout << getElement(i, j) << " ";
        }
        std::cout << std::endl;
    }
    std::cout<<"\n";
}

template<typename T>
T& Matice<T>::operator()(size_t row, size_t col) {
    if (row >= rows || col >= cols) {
        std::cout<<"index mimo rozmery matice";
        exit(0);
    }
    return dta[row][col];
}

template<typename T>
T Matice<T>::operator()(size_t row, size_t col) const{
    if (row >= rows || col >= cols) {
        std::cout<<"index mimo rozmery matice";
        exit(0);
    }
    return dta[row][col];
}


