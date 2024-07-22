#include "optimizer.h"

void unary_constant_folding_test() {
  Instruction *c = CreateConstant(5);

  Instruction *abs = CreateUnary(kAbs, c);
  Instruction *cos = CreateUnary(kCos, c);
  Instruction *exp = CreateUnary(kExp, c);
  Instruction *log = CreateUnary(kLog, c);
  Instruction *neg = CreateUnary(kNegate, c);
  Instruction *sin = CreateUnary(kSin, c);
  Instruction *tan = CreateUnary(kTan, c);
  Instruction *tanh = CreateUnary(kTanh, c);

  Optimizer opt;
  opt.Run(abs);
  assert(IsConstantWithValue(abs, 5));
  opt.Run(cos);
  assert(IsConstantWithValue(cos, std::cos(5)));
  opt.Run(exp);
  assert(IsConstantWithValue(exp, std::exp(5)));
  opt.Run(log);
  assert(IsConstantWithValue(log, std::log(5)));
  opt.Run(neg);
  assert(IsConstantWithValue(neg, -5));
  opt.Run(sin);
  assert(IsConstantWithValue(sin, std::sin(5)));
  opt.Run(tan);
  assert(IsConstantWithValue(tan, std::tan(5)));
  opt.Run(tanh);
  assert(IsConstantWithValue(tanh, std::tanh(5)));
}

void binary_constant_folding_test() {
  Instruction *c1 = CreateConstant(2);
  Instruction *c2 = CreateConstant(3);

  Instruction *add = CreateBinary(kAdd, c1, c2);
  Instruction *div = CreateBinary(kDivide, c1, c2);
  Instruction *max = CreateBinary(kMaximum, c1, c2);
  Instruction *min = CreateBinary(kMinimum, c1, c2);
  Instruction *mul = CreateBinary(kMultiply, c1, c2);
  Instruction *pow = CreateBinary(kPower, c1, c2);
  Instruction *sub = CreateBinary(kSubtract, c1, c2);

  Optimizer opt;
  opt.Run(add);
  assert(IsConstantWithValue(add, 5));
  opt.Run(div);
  assert(IsConstantWithValue(div, 2 / 3.0));
  opt.Run(max);
  assert(IsConstantWithValue(max, 3));
  opt.Run(min);
  assert(IsConstantWithValue(min, 2));
  opt.Run(mul);
  assert(IsConstantWithValue(mul, 6));
  opt.Run(pow);
  assert(IsConstantWithValue(pow, 8));
  opt.Run(sub);
  assert(IsConstantWithValue(sub, -1));
}

void binary_canonicalization_test() {
  Instruction *x = CreateParameter();
  Instruction *c = CreateConstant(3);

  Instruction *add = CreateBinary(kAdd, x, c);
  Instruction *mul = CreateBinary(kMultiply, x, c);

  Optimizer opt;
  opt.Run(add);
  assert(add->operand(0)->opcode() == kConstant &&
         add->operand(1)->opcode() == kParameter);
  opt.Run(mul);
  assert(mul->operand(0)->opcode() == kConstant &&
         mul->operand(1)->opcode() == kParameter);
}

void fold_add_0_test() {
  Instruction *x = CreateParameter();
  Instruction *c = CreateConstant(0);

  Instruction *add = CreateBinary(kAdd, c, x);

  Optimizer opt;
  opt.Run(add);
  assert(add->opcode() == kParameter);
}

void multiply_zero_test() {
  Instruction *c = CreateConstant(0);
  Instruction *x = CreateParameter();

  Instruction *mul = CreateBinary(kMultiply, c, x);

  Optimizer opt;
  opt.Run(mul);
  assert(IsConstantWithValue(mul, 0));
}

void multiply_one_test() {
  Instruction *c = CreateConstant(1);
  Instruction *x = CreateParameter();

  Instruction *mul = CreateBinary(kMultiply, c, x);

  Optimizer opt;
  opt.Run(mul);
  assert(mul->opcode() == kParameter);
}

void abs_negate_test() {
  Instruction *x = CreateParameter();

  Instruction *neg = CreateUnary(kNegate, x);
  Instruction *abs = CreateUnary(kAbs, neg);

  Optimizer opt;
  opt.Run(abs);
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
  opt.Run(add);
  assert(add->opcode() == kMultiply && add->operand(1)->opcode() == kAdd);
}

void add_to_multiply_test() {
  Instruction *x = CreateParameter();

  Instruction *add = CreateBinary(kAdd, x, x);

  Optimizer opt;
  opt.Run(add);
  assert(add->opcode() == kMultiply);
}

void divide_divide_test() {
  Instruction *x = CreateParameter();
  Instruction *y = CreateParameter();
  Instruction *z = CreateParameter();

  Instruction *div_lhs = CreateBinary(kDivide, CreateBinary(kDivide, x, y), z);
  Instruction *div_rhs = CreateBinary(kDivide, x, CreateBinary(kDivide, y, z));

  Optimizer opt;
  opt.Run(div_lhs);
  opt.Run(div_rhs);
  assert(div_lhs->operand(1)->opcode() == kMultiply &&
         div_rhs->operand(0)->opcode() == kMultiply);
}

void exp_log_test() {
  Instruction *x = CreateParameter();

  Instruction *exp_log = CreateUnary(kExp, CreateUnary(kLog, x));
  Instruction *log_exp = CreateUnary(kLog, CreateUnary(kExp, x));

  Optimizer opt;
  opt.Run(exp_log);
  opt.Run(log_exp);
  assert(exp_log->opcode() == kParameter && log_exp->opcode() == kParameter);
}

void multiply_abs_test() {
  Instruction *x = CreateParameter();
  Instruction *y = CreateParameter();

  Instruction *abs1 = CreateUnary(kAbs, x);
  Instruction *abs2 = CreateUnary(kAbs, y);
  Instruction *mul = CreateBinary(kMultiply, abs1, abs2);

  Optimizer opt;
  opt.Run(mul);
  assert(mul->opcode() == kAbs && mul->operand(0)->opcode() == kMultiply);
}

void multiply_exp_to_exp_add_test() {
  Instruction *x = CreateParameter();
  Instruction *y = CreateParameter();

  Instruction *exp_x = CreateUnary(kExp, x);
  Instruction *exp_y = CreateUnary(kExp, y);
  Instruction *mul = CreateBinary(kMultiply, exp_x, exp_y);

  Optimizer opt;
  opt.Run(mul);
  assert(mul->opcode() == kExp && mul->operand(0)->opcode() == kAdd);
}

void multiply_power_common_exponent_test() {
  Instruction *x = CreateParameter();
  Instruction *y = CreateParameter();
  Instruction *z = CreateParameter();

  Instruction *pow1 = CreateBinary(kPower, x, z);
  Instruction *pow2 = CreateBinary(kPower, y, z);
  Instruction *mul = CreateBinary(kMultiply, pow1, pow2);

  Optimizer opt;
  opt.Run(mul);
  assert(mul->opcode() == kPower && mul->operand(0)->opcode() == kMultiply);
}

void multiply_power_common_base_test() {
  Instruction *x = CreateParameter();
  Instruction *y = CreateParameter();
  Instruction *z = CreateParameter();

  Instruction *pow1 = CreateBinary(kPower, x, y);
  Instruction *pow2 = CreateBinary(kPower, x, z);
  Instruction *mul = CreateBinary(kMultiply, pow1, pow2);

  Optimizer opt;
  opt.Run(mul);
  assert(mul->opcode() == kPower && mul->operand(1)->opcode() == kAdd);
}

void double_negation_test() {
  Instruction *x = CreateParameter();

  Instruction *neg_x = CreateUnary(kNegate, x);
  Instruction *neg_neg_x = CreateUnary(kNegate, neg_x);

  Optimizer opt;
  opt.Run(neg_neg_x);
  assert(neg_neg_x->opcode() == kParameter);
}

void negate_subtract_folding_test() {
  Instruction *x = CreateParameter();
  Instruction *y = CreateParameter();

  Instruction *sub = CreateBinary(kSubtract, x, y);
  Instruction *neg = CreateUnary(kNegate, sub);

  Optimizer opt;
  opt.Run(neg);
  assert(neg->opcode() == kSubtract);
}

void pow_zero_test() {
  Instruction *x = CreateParameter();
  Instruction *c = CreateConstant(0);

  Instruction *pow = CreateBinary(kPower, x, c);

  Optimizer opt;
  opt.Run(pow);
  assert(IsConstantWithValue(pow, 1));
}

void pow_one_test() {
  Instruction *x = CreateParameter();
  Instruction *c = CreateConstant(1);

  Instruction *pow = CreateBinary(kPower, x, c);

  Optimizer opt;
  opt.Run(pow);
  assert(pow->opcode() == kParameter);
}

void square_test() {
  Instruction *x = CreateParameter();
  Instruction *c = CreateConstant(2);

  Instruction *pow = CreateBinary(kPower, x, c);

  Optimizer opt;
  opt.Run(pow);
  assert(pow->opcode() == kMultiply);
}

void zero_pow_test() {
  Instruction *c = CreateConstant(0);
  Instruction *x = CreateParameter();

  Instruction *pow = CreateBinary(kPower, c, x);

  Optimizer opt;
  opt.Run(pow);
  assert(IsConstantWithValue(pow, 0));
}

void one_pow_test() {
  Instruction *c = CreateConstant(1);
  Instruction *x = CreateParameter();

  Instruction *pow = CreateBinary(kPower, c, x);

  Optimizer opt;
  opt.Run(pow);
  assert(IsConstantWithValue(pow, 1));
}

void power_power_test() {
  Instruction *x = CreateParameter();
  Instruction *y = CreateParameter();
  Instruction *z = CreateParameter();

  Instruction *pow1 = CreateBinary(kPower, x, y);
  Instruction *pow2 = CreateBinary(kPower, pow1, z);

  Optimizer opt;
  opt.Run(pow2);
  assert(pow2->operand(1)->opcode() == kMultiply);
}

void subtract_logs_to_log_divide_test() {
  Instruction *x = CreateParameter();
  Instruction *y = CreateParameter();

  Instruction *log_x = CreateUnary(kLog, x);
  Instruction *log_y = CreateUnary(kLog, y);
  Instruction *sub = CreateBinary(kSubtract, log_x, log_y);

  Optimizer opt;
  opt.Run(sub);
  assert(sub->opcode() == kLog && sub->operand(0)->opcode() == kDivide);
}

int main() {
  unary_constant_folding_test();
  binary_constant_folding_test();
  binary_canonicalization_test();
  fold_add_0_test();
  multiply_zero_test();
  multiply_one_test();
  abs_negate_test();
  factor_add_multiply_test();
  add_to_multiply_test();
  divide_divide_test();
  exp_log_test();
  multiply_abs_test();
  multiply_exp_to_exp_add_test();
  multiply_power_common_exponent_test();
  multiply_power_common_base_test();
  double_negation_test();
  negate_subtract_folding_test();
  pow_zero_test();
  pow_one_test();
  square_test();
  zero_pow_test();
  one_pow_test();
  power_power_test();
  subtract_logs_to_log_divide_test();
  return 0;
}
