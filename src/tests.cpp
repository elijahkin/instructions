#include <iostream>

#include "optimizer.h"

void sigmoid_test() {
  Instruction *x = CreateVariable();
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
  Instruction *a = CreateVariable();
  Instruction *b = CreateVariable();
  Instruction *c = CreateVariable();

  Instruction *ab = CreateBinary(kMultiply, a, b);
  Instruction *ac = CreateBinary(kMultiply, a, c);

  Instruction *add = CreateBinary(kAdd, ab, ac);
  Instruction *exp = CreateUnary(kExp, add);

  Optimizer opt;
  opt.Optimize(add);
  std::cout << exp->to_string() << std::endl << std::endl;
}

void subtract_test() {
  Instruction *a = CreateVariable();
  Instruction *b = CreateVariable();
  Instruction *c = CreateVariable();

  Instruction *sub = CreateBinary(kSubtract, a, b);
  Instruction *neg = CreateUnary(kNegate, sub);

  Instruction *add = CreateBinary(kAdd, neg, c);

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
