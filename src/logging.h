#include <iostream>

#define MAX_VERBOSITY 10
#define VLOG(verbosity) Logger(__FILE__, __LINE__, verbosity)

class Logger {
private:
  const char *file_;
  int line_;
  int verbosity_;

public:
  Logger(const char *file, int line, int verbosity)
      : file_(file), line_(line), verbosity_(verbosity) {}

  Logger &operator<<(std::string msg) {
    if (verbosity_ <= MAX_VERBOSITY) {
      std::cout << "[" << file_ << ":" << line_ << "]: " << msg << std::endl;
    }
    return *this;
  }
};
