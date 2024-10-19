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


template<typename T>
void Matice<T>::shuffle_radky() {
    if (rows <= 1) {
        return;
    }
    std::random_device rd;
    std::mt19937 gen(rd());

    std::shuffle(dta, dta + rows, gen);
}

template<typename T>
void Matice<T>::shuffle_radkyavec(std::vector<T>& vec) {
    if (vec.size() != rows) {
        throw std::invalid_argument("Délka vektoru musí odpovídat počtu řádků matice.");
    }
    if (rows <= 1) {
        return;
    }
    std::random_device rd;
    std::mt19937 gen(rd());
    std::vector<size_t> indexy(rows);
    for (size_t i = 0; i < rows; ++i) {
        indexy[i] = i;
    }
    std::shuffle(indexy.begin(), indexy.end(), gen);
    T** new_dta = new T*[rows];
    std::vector<T> new_vec(rows);

    for (size_t i = 0; i < rows; ++i) {
        new_dta[i] = dta[indexy[i]];
        new_vec[i] = vec[indexy[i]];
    }

    for (size_t i = 0; i < rows; ++i) {
        dta[i] = new_dta[i];
        vec[i] = new_vec[i];
    }

    delete[] new_dta;
}

template<typename T>
void Matice<T>::rand_vypln(double min, double max){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(min,max);
    for (int i = 0;i<rows;i++){
        for (int j = 0;j<cols;j++){
            dta[i][j] = dis(gen);
        }
    }
}


template<typename T>
void Matice<T>::resize(size_t radky, size_t sloupce){
    for (size_t i = 0; i < rows; ++i) {
        delete[] dta[i];
    }
    delete[] dta;

    rows = radky;
    cols = sloupce;
    dta = new T*[rows];
    for (size_t i = 0; i < rows; ++i) {
        dta[i] = new T[cols]();
    }
}

template<typename T>
void Matice<T>::o180_nuly(size_t layers) {
    size_t new_rows = rows + 2 * layers;
    size_t new_cols = cols + 2 * layers;

    T** new_dta = new T*[new_rows];
    for (size_t i = 0; i < new_rows; ++i) {
        new_dta[i] = new T[new_cols]();
    }

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            new_dta[new_rows - layers - 1 - i][new_cols - layers - 1 - j] = dta[i][j];
        }
    }

    for (size_t i = 0; i < rows; ++i) {
        delete[] dta[i];
    }
    delete[] dta;

    rows = new_rows;
    cols = new_cols;
    dta = new_dta;
}