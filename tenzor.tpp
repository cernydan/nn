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
void Tenzor<T>::set_matrix(size_t depth, const Matice<T>& matice) {
    if (depth >= this->depth){
        std::cout << "Index depth větší než depth tenzoru." << std::endl;
        exit(0);
    }

    if (matice.getRows() != rows || matice.getCols() != cols) {
            std::cout << "Rozměry matice neodpovídají rozměrům tenzoru." << std::endl;
            exit(0);
        }

    for (int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){
            dta[depth][i][j] = matice(i,j);
        }
    }
    
}

template<typename T>
void Tenzor<T>::printRozmery() {
    std::cout<<depth<<" "<<rows<<" "<<cols;
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

template<typename T>
void Tenzor<T>::rand_vypln(double mean, double sd){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dis(mean,sd);
    for(int k = 0;k<depth;k++){
        for (int i = 0;i<rows;i++){
            for (int j = 0;j<cols;j++){
                dta[k][i][j] = dis(gen);
            }
        }
    }
}

template<typename T>
void Tenzor<T>::flip180() {
    for (size_t d = 0; d < depth; ++d) {
        for (size_t r = 0; r < rows / 2; ++r) {
            for (size_t c = 0; c < cols; ++c) {
                std::swap(dta[d][r][c], dta[d][rows - 1 - r][cols - 1 - c]);
            }
        }
        if (rows % 2 != 0) {
            size_t middle_row = rows / 2;
            for (size_t c = 0; c < cols / 2; ++c) {
                std::swap(dta[d][middle_row][c], dta[d][middle_row][cols - 1 - c]);
            }
        }
    }
}

template<typename T>
void Tenzor<T>::obal_nul(size_t layers) {
    size_t new_rows = rows + 2 * layers;
    size_t new_cols = cols + 2 * layers;

    T*** new_dta = new T**[depth];
    for (size_t d = 0; d < depth; ++d) {
        new_dta[d] = new T*[new_rows];
        for (size_t r = 0; r < new_rows; ++r) {
            new_dta[d][r] = new T[new_cols];
            for (size_t c = 0; c < new_cols; ++c) {
                if (r < layers || r >= rows + layers || c < layers || c >= cols + layers) {
                    new_dta[d][r][c] = 0;
                } else {
                    new_dta[d][r][c] = dta[d][r - layers][c - layers];
                }
            }
        }
    }
    for (size_t d = 0; d < depth; ++d) {
        for (size_t r = 0; r < rows; ++r) {
            delete[] dta[d][r];
        }
        delete[] dta[d];
    }
    delete[] dta;

    rows = new_rows;
    cols = new_cols;
    dta = new_dta;
}

template<typename T>
void Tenzor<T>::dilace(size_t row_pad, size_t col_pad) {
    size_t new_rows = rows + (rows - 1) * row_pad; 
    size_t new_cols = cols + (cols - 1) * col_pad;

    T*** new_data = new T**[depth];
    for (size_t d = 0; d < depth; ++d) {
        new_data[d] = new T*[new_rows];
        for (size_t r = 0; r < new_rows; ++r) {
            new_data[d][r] = new T[new_cols]();
        }
    }

    for (size_t d = 0; d < depth; ++d) {
        for (size_t r = 0; r < rows; ++r) {
            for (size_t c = 0; c < cols; ++c) {
                new_data[d][r * (row_pad + 1)][c * (col_pad + 1)] = dta[d][r][c];
            }
        }
    }

    for (size_t d = 0; d < depth; ++d) {
        for (size_t r = 0; r < rows; ++r) {
            delete[] dta[d][r];
        }
        delete[] dta[d];
    }
    delete[] dta;

    rows = new_rows;
    cols = new_cols;
    dta = new_data;
}

template<typename T>
void Tenzor<T>::resize(size_t hloubka, size_t radky, size_t sloupce) {
    // Uvolnění staré paměti
    if (dta != nullptr) {
        for (size_t i = 0; i < depth; ++i) {
            for (size_t j = 0; j < rows; ++j) {
                delete[] dta[i][j];
            }
            delete[] dta[i];
        }
        delete[] dta;
    }

    // Nastavení nových rozměrů
    depth = hloubka;
    rows = radky;
    cols = sloupce;

    // Alokace nové paměti
    dta = new T**[depth];
    for (size_t i = 0; i < depth; ++i) {
        dta[i] = new T*[rows];
        for (size_t j = 0; j < rows; ++j) {
            dta[i][j] = new T[cols]();
        }
    }
}

