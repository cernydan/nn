#include "tenzor.h"

template<typename T>
Tenzor<T>::Tenzor() :     // Konstruktor bez rozměrů
    dta(nullptr), 
    depth(0), 
    rows(0), 
    cols(0) {}

template<typename T>
Tenzor<T>::Tenzor(size_t depth, size_t rows, size_t cols) :     // Konstruktor s rozměry
    depth(depth), 
    rows(rows), 
    cols(cols) {
    dta = new T**[depth];
    for (size_t i = 0; i < depth; ++i) {
        dta[i] = new T*[rows];
        for (size_t j = 0; j < rows; ++j) {
            dta[i][j] = new T[cols]();
        }
    }
}

template<typename T>
Tenzor<T>::~Tenzor() {        // Destruktor
    for (size_t i = 0; i < depth; ++i) {
        for (size_t j = 0; j < rows; ++j) {
            delete[] dta[i][j];
        }
        delete[] dta[i];
    }
    delete[] dta;
}

template<typename T>
Tenzor<T>::Tenzor(const Tenzor& other) :         // Kopírovací konstruktor
    depth(other.depth), 
    rows(other.rows), 
    cols(other.cols) {
    dta = new T**[depth];
    for (size_t i = 0; i < depth; ++i) {
        dta[i] = new T*[rows];
        for (size_t j = 0; j < rows; ++j) {
            dta[i][j] = new T[cols];
            std::copy(other.dta[i][j], other.dta[i][j] + cols, dta[i][j]);
        }
    }
}

template<typename T>
Tenzor<T>::Tenzor(Tenzor&& other) noexcept :     // Move konstruktor
    dta(other.dta), 
    depth(other.depth), 
    rows(other.rows), 
    cols(other.cols) {
    other.dta = nullptr;
    other.depth = 0;
    other.rows = 0;
    other.cols = 0;
}

template<typename T>
Tenzor<T>& Tenzor<T>::operator=(const Tenzor& other) {    // Kopírovací přiřazovací operátor
    if (this != &other) {
        for (size_t i = 0; i < depth; ++i) {
            for (size_t j = 0; j < rows; ++j) {
                delete[] dta[i][j];
            }
            delete[] dta[i];
        }
        delete[] dta;

        depth = other.depth;
        rows = other.rows;
        cols = other.cols;

        dta = new T**[depth];
        for (size_t i = 0; i < depth; ++i) {
            dta[i] = new T*[rows];
            for (size_t j = 0; j < rows; ++j) {
                dta[i][j] = new T[cols];
                std::copy(other.dta[i][j], other.dta[i][j] + cols, dta[i][j]);
            }
        }
    }
    return *this;
}

template<typename T>
Tenzor<T>& Tenzor<T>::operator=(Tenzor&& other) noexcept {    // Move přiřazovací operátor
    if (this != &other) {
        for (size_t i = 0; i < depth; ++i) {
            for (size_t j = 0; j < rows; ++j) {
                delete[] dta[i][j];
            }
            delete[] dta[i];
        }
        delete[] dta;

        dta = other.dta;
        depth = other.depth;
        rows = other.rows;
        cols = other.cols;

        other.dta = nullptr;
        other.depth = 0;
        other.rows = 0;
        other.cols = 0;
    }
    return *this;
}

template<typename T>
void Tenzor<T>::load_stdvv(const std::vector<std::vector<std::vector<T>>>& input) {
    if (!input.empty()) {
        depth = input.size();
        rows = input[0].size();
        cols = input[0][0].size();
        delete[] dta;

        dta = new T**[depth];
        for (size_t i = 0; i < depth; ++i) {
            dta[i] = new T*[rows];
            for (size_t j = 0; j < rows; ++j) {
                dta[i][j] = new T[cols];
                if (input[i][j].size() != cols) {
                    std::cout << "rozmery vstupu neodpovidaji";
                    exit(0);
                }
                std::copy(input[i][j].begin(), input[i][j].end(), dta[i][j]);
            }
        }
    }
}

template<typename T>
size_t Tenzor<T>::getDepth() const{
    return depth;
}

template<typename T>
size_t Tenzor<T>::getRows() const{
    return rows;
}

template<typename T>
size_t Tenzor<T>::getCols() const{
    return cols;
}

template<typename T>
T Tenzor<T>::getElement(size_t depth, size_t row, size_t col) {
    if (depth >= this->depth || row >= rows || col >= cols) {
        std::cout << "index mimo rozmery";
        exit(0);
    }
    return dta[depth][row][col];
}

template<typename T>
void Tenzor<T>::setElement(size_t depth, size_t row, size_t col, T value) {
    if (depth >= this->depth || row >= rows || col >= cols) {
        std::cout << "index mimo rozmery";
        exit(0);
    }
    dta[depth][row][col] = value;
}

template<typename T>
void Tenzor<T>::printTenzor() {
    for (size_t i = 0; i < depth; ++i) {
        std::cout << "Vrstva " << i << ":\n";
        for (size_t j = 0; j < rows; ++j) {
            for (size_t k = 0; k < cols; ++k) {
                std::cout << dta[i][j][k] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

template<typename T>
T& Tenzor<T>::operator()(size_t depth, size_t row, size_t col) {
    if (depth >= this->depth || row >= rows || col >= cols) {
        std::cout << "index mimo rozmery tenzoru";
        exit(0);
    }
    return dta[depth][row][col];
}

template<typename T>
T Tenzor<T>::operator()(size_t depth, size_t row, size_t col) const {
    if (depth >= this->depth || row >= rows || col >= cols) {
        std::cout << "index mimo rozmery tenzoru";
        exit(0);
    }
    return dta[depth][row][col];
}

template<typename T>
void Tenzor<T>::add_matrix(const Matice<T>& matice) {
    // kdyz prvni matice, nastavi rozmery, jinak kontroluje rozmery
    if (depth == 0) {
        rows = matice.getRows();
        cols = matice.getCols();
    } else {
        if (matice.getRows() != rows || matice.getCols() != cols) {
            std::cout << "Rozměry nové matice neodpovídají rozměrům tenzoru." << std::endl;
            exit(0);
        }
    }

    // Vytvoření nového prostoru pro tenzor s větší hloubkou
    T*** newData = new T**[depth + 1];

    // Překopírování stávajících dat do nového pole
    for (size_t i = 0; i < depth; ++i) {
        newData[i] = dta[i];
    }

    // Přidání nové matice do poslední vrstvy nového tenzoru
    newData[depth] = new T*[rows];
    for (size_t j = 0; j < rows; ++j) {
        newData[depth][j] = new T[cols];
        std::copy(matice.dta[j], matice.dta[j] + cols, newData[depth][j]);
    }

    // Uvolnění původního prostoru, ale nealokuje se dta[i], protože jsme si zachovali ukazatele
    delete[] dta;

    // Přiřazení nově vytvořeného pole
    dta = newData;

    // Zvýšení hloubky tenzoru
    ++depth;
}

