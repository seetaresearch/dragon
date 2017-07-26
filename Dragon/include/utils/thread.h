// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_UTILS_THREAD_H_
#define DRAGON_UTILS_THREAD_H_

#include <memory>
#include <mutex>
#include <condition_variable>
#include <thread>

namespace dragon {

struct thread_interrupted {};

class InterruptionPoint {
 public:
    InterruptionPoint() : stop(false) {}
    void Interrupt() {
        std::unique_lock<std::mutex> lock(mutex);
        stop = true;
    }
    void InterruptionRequested() {
        std::unique_lock<std::mutex> lock(mutex);
        if (stop) throw thread_interrupted();
    }

 protected:
    bool stop;
    std::mutex mutex;
    std::condition_variable cond;
};

class BaseThread {
 public:
    ~BaseThread() { Stop(); }
    void Start() {
        thread = std::unique_ptr<std::thread>(
            new std::thread(std::bind(&BaseThread::ThreadRun, this)));
    }
    void Stop() {
        interruption_point.Interrupt();
        thread->join();
    }
    bool must_stop() {
        interruption_point.InterruptionRequested();
        return false;
    }

 protected:
    virtual void ThreadRun() {
        try {
            while (!must_stop()) {}
        } catch (const thread_interrupted&) {}
    }

 private:
    std::unique_ptr<std::thread> thread;
    InterruptionPoint interruption_point;
};

}    // namespace dragon

#endif    // DRAGON_UTILS_THREAD_H_