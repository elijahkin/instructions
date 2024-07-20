#include <iostream>

#include "instruction.h"

int main() {
  // Create a sigmoid function as a test
  Instruction *x = CreateNullary(kVariable);
  Instruction *neg = CreateUnary(kNegate, x);
  Instruction *exp = CreateUnary(kExp, neg);
  Instruction *one = CreateNullary(kConstant);
  Instruction *add = CreateBinary(kAdd, one, exp);
  Instruction *div = CreateBinary(kDivide, one, add);

  assert(x->arity() == 0);
  assert(neg->arity() == 1);
  assert(exp->arity() == 1);
  assert(one->arity() == 0);
  assert(add->arity() == 2);
  assert(div->arity() == 2);

  std::cout << div->to_string() << std::endl;
  return 0;
}
