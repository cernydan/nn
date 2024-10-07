// #include "threadpool.h"

// // Konstruktor - inicializace poolu s daným počtem vláken
// ThreadPool::ThreadPool(size_t numThreads) : stop(false) {
//     for (size_t i = 0; i < numThreads; ++i) {
//         workers.emplace_back([this] {
//             worker();
//         });
//     }
// }

// // Destruktor - čeká na dokončení všech vláken a uvolňuje zdroje
// ThreadPool::~ThreadPool() {
//     {
//         std::unique_lock<std::mutex> lock(queue_mutex);
//         stop = true;
//     }
//     condition.notify_all();  // Oznám všem vláknům, že mají skončit

//     for (std::thread &worker : workers) {
//         worker.join();  // Připoj všechna vlákna k hlavnímu vláknu
//     }
// }

// // enqueueTask pro přidání úkolu do fronty a získání výsledku jako future
// template<class F, class... Args>
// auto ThreadPool::enqueueTask(F&& f, Args&&... args)
//     -> std::future<typename std::result_of<F(Args...)>::type> {

//     using return_type = typename std::result_of<F(Args...)>::type;

//     // Zabalíme úkol (task) do std::packaged_task, což nám umožní získat std::future
//     auto task = std::make_shared<std::packaged_task<return_type()>>(
//         std::bind(std::forward<F>(f), std::forward<Args>(args)...)
//     );

//     std::future<return_type> res = task->get_future();  // Získáme future pro výsledek úkolu
//     {
//         std::unique_lock<std::mutex> lock(queue_mutex);

//         // Kontrola, jestli je pool zastaven
//         if (stop) {
//             throw std::runtime_error("enqueue on stopped ThreadPool");
//         }

//         // Přidáme úkol do fronty
//         tasks.emplace([task]() { (*task)(); });
//     }

//     condition.notify_one();  // Upozorníme vlákno, že má nový úkol
//     return res;
// }

// // Funkce, kterou vykonává každé vlákno
// void ThreadPool::worker() {
//     while (true) {
//         std::function<void()> task;

//         {
//             std::unique_lock<std::mutex> lock(queue_mutex);

//             // Čekej, dokud nejsou úkoly k dispozici nebo dokud nemáme zastavit
//             condition.wait(lock, [this] {
//                 return stop || !tasks.empty();
//             });

//             if (stop && tasks.empty()) {
//                 return;  // Ukonči vlákno, pokud má pool zastavit a nejsou úkoly
//             }

//             // Získej úkol z fronty
//             task = std::move(tasks.front());
//             tasks.pop();
//         }

//         // Vykonej úkol
//         task();
//     }
// }