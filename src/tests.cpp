#include "optimizer.h"

void factor_add_multiply_test() {
  Instruction *x = CreateParameter();
  Instruction *y = CreateParameter();
  Instruction *z = CreateParameter();

  Instruction *lhs = CreateBinary(kMultiply, x, y);
  Instruction *rhs = CreateBinary(kMultiply, x, z);
  Instruction *add = CreateBinary(kAdd, lhs, rhs);

  Optimizer opt;
  opt.Optimize(add);
  assert(add->opcode() == kMultiply && add->operand(1)->opcode() == kAdd);
}

void add_to_multiply_test() {
  Instruction *x = CreateParameter();

  Instruction *add = CreateBinary(kAdd, x, x);

  Optimizer opt;
  opt.Optimize(add);
  assert(add->opcode() == kMultiply);
}

void exp_log_test() {
  Instruction *x = CreateParameter();
  Instruction *y = CreateParameter();

  Instruction *log_x = CreateUnary(kLog, x);
  Instruction *exp_log_x = CreateUnary(kExp, log_x);
  Instruction *exp_y = CreateUnary(kExp, y);
  Instruction *log_exp_y = CreateUnary(kLog, exp_y);
  Instruction *add = CreateBinary(kAdd, exp_log_x, log_exp_y);

  Optimizer opt;
  opt.Optimize(add);
  assert(add->operand(0)->opcode() == kParameter &&
         add->operand(1)->opcode() == kParameter);
}

void multiply_exp_to_exp_add_test() {
  Instruction *x = CreateParameter();
  Instruction *y = CreateParameter();

  Instruction *exp_x = CreateUnary(kExp, x);
  Instruction *exp_y = CreateUnary(kExp, y);
  Instruction *mul = CreateBinary(kMultiply, exp_x, exp_y);

  Optimizer opt;
  opt.Optimize(mul);
  assert(mul->opcode() == kExp && mul->operand(0)->opcode() == kAdd);
}

void double_negation_test() {
  Instruction *x = CreateParameter();

  Instruction *neg_x = CreateUnary(kNegate, x);
  Instruction *neg_neg_x = CreateUnary(kNegate, neg_x);

  Optimizer opt;
  opt.Optimize(neg_neg_x);
  assert(neg_neg_x->opcode() == kParameter);
}

void negate_subtract_folding_test() {
  Instruction *x = CreateParameter();
  Instruction *y = CreateParameter();

  Instruction *sub = CreateBinary(kSubtract, x, y);
  Instruction *neg = CreateUnary(kNegate, sub);

  Optimizer opt;
  opt.Optimize(neg);
  assert(neg->opcode() == kSubtract);
}

void subtract_logs_to_log_divide_test() {
  Instruction *x = CreateParameter();
  Instruction *y = CreateParameter();

  Instruction *log_x = CreateUnary(kLog, x);
  Instruction *log_y = CreateUnary(kLog, y);
  Instruction *sub = CreateBinary(kSubtract, log_x, log_y);

  Optimizer opt;
  opt.Optimize(sub);
  assert(sub->opcode() == kLog && sub->operand(0)->opcode() == kDivide);
}

int main() {
  factor_add_multiply_test();
  add_to_multiply_test();
  exp_log_test();
  multiply_exp_to_exp_add_test();
  double_negation_test();
  negate_subtract_folding_test();
  subtract_logs_to_log_divide_test();
  return 0;
}
