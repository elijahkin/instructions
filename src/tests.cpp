#include "optimizer.h"

void add_to_multiply_test() {
  Instruction *x = CreateParameter();

  Instruction *add = CreateBinary(kAdd, x, x);
  Instruction *neg = CreateUnary(kNegate, add);

  Optimizer opt;
  opt.Optimize(add);
  assert(neg->operand(0)->opcode() == kMultiply);
}

void factor_add_multiply_test() {
  Instruction *x = CreateParameter();
  Instruction *y = CreateParameter();
  Instruction *z = CreateParameter();

  Instruction *lhs = CreateBinary(kMultiply, x, y);
  Instruction *rhs = CreateBinary(kMultiply, x, z);
  Instruction *add = CreateBinary(kAdd, lhs, rhs);
  Instruction *exp = CreateUnary(kExp, add);

  Optimizer opt;
  opt.Optimize(add);
  assert(exp->operand(0)->opcode() == kMultiply);
}

void negate_subtract_folding_test() {
  Instruction *x = CreateParameter();
  Instruction *y = CreateParameter();
  Instruction *z = CreateParameter();

  Instruction *sub = CreateBinary(kSubtract, x, y);
  Instruction *neg = CreateUnary(kNegate, sub);
  Instruction *add = CreateBinary(kAdd, neg, z);

  Optimizer opt;
  opt.Optimize(neg);
  assert(add->operand(0)->opcode() == kSubtract);
}

int main() {
  add_to_multiply_test();
  factor_add_multiply_test();
  negate_subtract_folding_test();
  return 0;
}
