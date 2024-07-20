#include <iostream>

#include "optimizer.h"

void sigmoid_test() {
  Instruction *x = CreateParameter();

  Instruction *neg = CreateUnary(kNegate, x);
  Instruction *exp = CreateUnary(kExp, neg);
  Instruction *one = CreateConstant(1);
  Instruction *add = CreateBinary(kAdd, one, exp);
  Instruction *div = CreateBinary(kDivide, one, add);

  assert(x->arity() == 0);
  assert(neg->arity() == 1);
  assert(exp->arity() == 1);
  assert(one->arity() == 0);
  assert(add->arity() == 2);
  assert(div->arity() == 2);

  std::cout << div->to_string() << std::endl << std::endl;
}

void rewrite_test() {
  Instruction *x = CreateParameter();
  Instruction *y = CreateParameter();
  Instruction *z = CreateParameter();

  Instruction *lhs = CreateBinary(kMultiply, x, y);
  Instruction *rhs = CreateBinary(kMultiply, x, z);
  Instruction *add = CreateBinary(kAdd, lhs, rhs);
  Instruction *exp = CreateUnary(kExp, add);

  Optimizer opt;
  opt.Optimize(add);
  std::cout << exp->to_string() << std::endl << std::endl;
}

void subtract_test() {
  Instruction *x = CreateParameter();
  Instruction *y = CreateParameter();
  Instruction *z = CreateParameter();

  Instruction *sub = CreateBinary(kSubtract, x, y);
  Instruction *neg = CreateUnary(kNegate, sub);
  Instruction *add = CreateBinary(kAdd, neg, z);

  Optimizer opt;
  opt.Optimize(neg);
  std::cout << add->to_string() << std::endl << std::endl;
}

int main() {
  sigmoid_test();
  rewrite_test();
  subtract_test();
  return 0;
}
