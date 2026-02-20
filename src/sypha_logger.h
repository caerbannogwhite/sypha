#ifndef SYPHA_LOGGER_H
#define SYPHA_LOGGER_H

#include <atomic>
#include <condition_variable>
#include <cstdarg>
#include <deque>
#include <mutex>
#include <string>
#include <thread>

enum SyphaLogLevel
{
    LOG_ERROR = 0,
    LOG_WARN = 1,
    LOG_INFO = 2,
    LOG_DEBUG = 3,
    LOG_TRACE = 4,
};

class SyphaLogger
{
public:
    explicit SyphaLogger(double originMs, SyphaLogLevel verbosity = LOG_INFO);
    ~SyphaLogger();

    SyphaLogger(const SyphaLogger &) = delete;
    SyphaLogger &operator=(const SyphaLogger &) = delete;

    void log(SyphaLogLevel level, const char *fmt, ...);
    void flush();

    void setHardTimeLimit(double limitMs);
    bool isStopRequested() const;
    void requestStop();

    SyphaLogLevel getVerbosity() const;
    void setVerbosity(SyphaLogLevel level);

private:
    struct LogEntry
    {
        SyphaLogLevel level;
        double timestampMs;
        std::string message;
    };

    void threadFunc();
    void drainQueue(std::deque<LogEntry> &batch);
    void writeEntry(const LogEntry &entry) const;
    double elapsedSeconds(double timestampMs) const;

    static const char *levelTag(SyphaLogLevel level);
    static const char *levelColor(SyphaLogLevel level);
    static double timerMs();

    double originMs_;
    SyphaLogLevel verbosity_;

    std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<LogEntry> queue_;
    bool shutdown_ = false;
    bool flushRequested_ = false;

    std::thread thread_;

    double hardTimeLimitMs_ = 0.0;
    std::atomic<bool> stopRequested_{false};

    std::mutex flushMutex_;
    std::condition_variable flushCv_;
    bool flushDone_ = false;
};

#endif // SYPHA_LOGGER_H
