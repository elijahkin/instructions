#include "optimizer.h"

void abs_negate_test() {
  Instruction *x = CreateParameter();

  Instruction *neg = CreateUnary(kNegate, x);
  Instruction *abs = CreateUnary(kAbs, neg);

  Optimizer opt;
  opt.Optimize(abs);
  assert(abs->operand(0)->opcode() == kParameter);
}

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

void divide_divide_test() {
  Instruction *x = CreateParameter();
  Instruction *y = CreateParameter();
  Instruction *z = CreateParameter();

  Instruction *div_lhs = CreateBinary(kDivide, CreateBinary(kDivide, x, y), z);
  Instruction *div_rhs = CreateBinary(kDivide, x, CreateBinary(kDivide, y, z));

  Optimizer opt;
  opt.Optimize(div_lhs);
  opt.Optimize(div_rhs);
  assert(div_lhs->operand(1)->opcode() == kMultiply &&
         div_rhs->operand(0)->opcode() == kMultiply);
}

void exp_log_test() {
  Instruction *x = CreateParameter();

  Instruction *exp_log = CreateUnary(kExp, CreateUnary(kLog, x));
  Instruction *log_exp = CreateUnary(kLog, CreateUnary(kExp, x));

  Optimizer opt;
  opt.Optimize(exp_log);
  opt.Optimize(log_exp);
  assert(exp_log->opcode() == kParameter && log_exp->opcode() == kParameter);
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
  abs_negate_test();
  factor_add_multiply_test();
  add_to_multiply_test();
  divide_divide_test();
  exp_log_test();
  multiply_exp_to_exp_add_test();
  double_negation_test();
  negate_subtract_folding_test();
  subtract_logs_to_log_divide_test();
  return 0;
}
