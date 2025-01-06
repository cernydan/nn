    enum Co {radky,lag};
    std::vector<std::vector<LSTMNeuron>> lstm_sit;
    std::vector<std::vector<double>> test_data;

    std::vector<Matice<double>> kernely_v2d;

    Matice<double> max_pool(Matice<double> vstupnim, size_t oknorad, size_t oknosl);
    Matice<double> max_pool_fullstep(Matice<double> vstupnim, size_t oknorad, size_t oknosl);
    Tenzor<double> max_pool_fullstep_3d(Tenzor<double> vstupnim, size_t oknorad, size_t oknosl);
    Matice<double> avg_pool(Matice<double> vstupnim, size_t oknorad, size_t oknosl);

    Matice<double> udelej_api(int n, double beta, Co coze, int kolik);
    Matice<double> udelej_prumery(int n, Co coze, int kolik);

    void init_lstm(int poc_vstupu, const std::vector<int>& rozmers);
    void online_lstm(int iter);
    void lstm_1cell(int batch_size, int iter);

    void online_bp_thread(int iter);
    void online_bp_th(int iter);
