#pragma once
#ifndef THREADPOOL_H
#define THREADPOOL_H

#include <iostream>
#include <vector>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>
#include <stdexcept>

class ThreadPool {
public:
    ThreadPool(size_t numThreads);  // Konstruktor
    ~ThreadPool();  // Destruktor

    // enqueueTask pro přidání úkolu do fronty a získání výsledku jako future
    template<class F, class... Args>
    auto enqueueTask(F&& f, Args&&... args)
        -> std::future<typename std::result_of<F(Args...)>::type>;

private:
    // Funkce, kterou vykonává každé vlákno
    void worker();

    // Vlákna
    std::vector<std::thread> workers;

    // Fronta úkolů
    std::queue<std::function<void()>> tasks;

    // Synchronizace
    std::mutex queue_mutex;
    std::condition_variable condition;

    // Indikace, zda pool funguje nebo byl zastaven
    bool stop;
};



#endif // THREADPOOL_H