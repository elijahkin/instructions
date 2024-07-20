#include <iostream>

#define LOG(level) Logger(__FILE__, __LINE__, level)

class Logger {
private:
  const char *file_;
  int line_;
  int level_;

public:
  Logger(const char *file, int line, int level)
      : file_(file), line_(line), level_(level) {}

  Logger &operator<<(std::string msg) {
    std::cout << file_ << "(" << line_ << "): " << msg;
    return *this;
  }

  ~Logger() { std::cout << std::endl; }
};
