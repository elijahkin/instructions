#include <vector>

class Shape {
private:
  std::vector<size_t> dimensions_;

public:
  Shape(std::vector<size_t> dimensions) { dimensions_ = dimensions; }
};
