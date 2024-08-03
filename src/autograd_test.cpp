#include "autograd.h"

int main() {
  Instruction* x = CreateParameter();
  Instruction* tan = CreateUnary(kTan, x);
  VLOG(10) << tan->to_string();

  Autograd grad;
  Instruction* deriv = grad.Derivative(tan);
  VLOG(10) << deriv->to_string();
}
