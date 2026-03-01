#include "sypha_logger.h"

#include <chrono>
#include <cstdio>
#include <cstring>

double SyphaLogger::timerMs()
{
    using namespace std::chrono;
    auto now = steady_clock::now();
    return duration<double, std::milli>(now.time_since_epoch()).count();
}

SyphaLogger::SyphaLogger(double originMs, SyphaLogLevel verbosity)
    : originMs_(originMs), verbosity_(verbosity)
{
    thread_ = std::thread(&SyphaLogger::threadFunc, this);
}

SyphaLogger::~SyphaLogger()
{
    flush();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        shutdown_ = true;
    }
    cv_.notify_one();
    if (thread_.joinable())
    {
        thread_.join();
    }
}

void SyphaLogger::log(SyphaLogLevel level, const char *fmt, ...)
{
    if (static_cast<int>(level) > static_cast<int>(verbosity_))
    {
        return;
    }

    char buf[1024];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);

    LogEntry entry;
    entry.level = level;
    entry.timestampMs = timerMs();
    entry.message = buf;

    {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push_back(std::move(entry));
    }
    cv_.notify_one();
}

void SyphaLogger::flush()
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty())
        {
            return;
        }
        flushRequested_ = true;
        flushDone_ = false;
    }
    cv_.notify_one();

    std::unique_lock<std::mutex> flock(flushMutex_);
    flushCv_.wait(flock, [this] { return flushDone_; });
}

void SyphaLogger::setHardTimeLimit(double limitMs)
{
    hardTimeLimitMs_ = limitMs;
}

bool SyphaLogger::isStopRequested() const
{
    return stopRequested_.load(std::memory_order_relaxed);
}

void SyphaLogger::requestStop()
{
    stopRequested_.store(true, std::memory_order_relaxed);
}

SyphaLogLevel SyphaLogger::getVerbosity() const
{
    return verbosity_;
}

void SyphaLogger::setVerbosity(SyphaLogLevel level)
{
    verbosity_ = level;
}

void SyphaLogger::threadFunc()
{
    std::deque<LogEntry> batch;

    while (true)
    {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait_for(lock, std::chrono::milliseconds(100),
                         [this] { return !queue_.empty() || shutdown_; });

            if (shutdown_ && queue_.empty())
            {
                break;
            }

            batch.swap(queue_);
            bool needFlushSignal = flushRequested_;
            flushRequested_ = false;

            lock.unlock();

            for (const auto &entry : batch)
            {
                writeEntry(entry);
            }
            batch.clear();
            fflush(stdout);

            if (needFlushSignal)
            {
                std::lock_guard<std::mutex> flock(flushMutex_);
                flushDone_ = true;
                flushCv_.notify_one();
            }
        }

        // Watchdog: check hard time limit
        if (hardTimeLimitMs_ > 0.0 && !stopRequested_.load(std::memory_order_relaxed))
        {
            double elapsedMs = timerMs() - originMs_;
            if (elapsedMs >= hardTimeLimitMs_)
            {
                stopRequested_.store(true, std::memory_order_relaxed);
            }
        }
    }
}

void SyphaLogger::drainQueue(std::deque<LogEntry> &batch)
{
    std::lock_guard<std::mutex> lock(mutex_);
    batch.swap(queue_);
}

void SyphaLogger::writeEntry(const LogEntry &entry) const
{
    static const char *resetColor = "\033[0m";
    fprintf(stdout, "%s[%8.3fs] [%-5s]%s %s\n",
            levelColor(entry.level),
            elapsedSeconds(entry.timestampMs),
            levelTag(entry.level),
            resetColor,
            entry.message.c_str());
}

double SyphaLogger::elapsedSeconds(double timestampMs) const
{
    return (timestampMs - originMs_) / 1000.0;
}

const char *SyphaLogger::levelTag(SyphaLogLevel level)
{
    switch (level)
    {
    case LOG_ERROR:
        return "ERROR";
    case LOG_WARN:
        return "WARN";
    case LOG_INFO:
        return "INFO";
    case LOG_DEBUG:
        return "DEBUG";
    case LOG_TRACE:
        return "TRACE";
    default:
        return "?????";
    }
}

const char *SyphaLogger::levelColor(SyphaLogLevel level)
{
    switch (level)
    {
    case LOG_ERROR:
        return "\033[1;31m";
    case LOG_WARN:
        return "\033[33m";
    case LOG_INFO:
        return "\033[32m";
    case LOG_DEBUG:
        return "\033[36m";
    case LOG_TRACE:
        return "\033[90m";
    default:
        return "";
    }
}
