/*!
 * Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
 *
 * Licensed under the BSD 2-Clause License.
 * You should have received a copy of the BSD 2-Clause License
 * along with the software. If not, See,
 *
 *     <https://opensource.org/licenses/BSD-2-Clause>
 *
 * ------------------------------------------------------------
 */

#ifndef DRAGON_UTILS_THREAD_H_
#define DRAGON_UTILS_THREAD_H_

#include <condition_variable>
#include <memory>
#include <mutex>
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
  ~BaseThread() {
    Stop();
  }

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
      while (!must_stop()) {
      }
    } catch (const thread_interrupted&) {
    }
  }

 private:
  std::unique_ptr<std::thread> thread;
  InterruptionPoint interruption_point;
};

} // namespace dragon

#endif // DRAGON_UTILS_THREAD_H_
