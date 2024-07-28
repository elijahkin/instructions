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

void even_negate_test() {
  Instruction *x = CreateParameter();

  Instruction *neg = CreateUnary(kNegate, x);
  Instruction *abs = CreateUnary(kAbs, neg);
  Instruction *cos = CreateUnary(kCos, neg);

  Optimizer opt;
  opt.Run(abs);
  assert(abs->opcode() == kAbs && abs->operand(0)->opcode() == kParameter);
  opt.Run(cos);
  assert(cos->opcode() == kCos && cos->operand(0)->opcode() == kParameter);
}

void odd_negate_test() {
  Instruction *x = CreateParameter();

  Instruction *neg = CreateUnary(kNegate, x);
  Instruction *sin = CreateUnary(kSin, neg);
  Instruction *tan = CreateUnary(kTan, neg);
  Instruction *tanh = CreateUnary(kTanh, neg);

  Optimizer opt;
  opt.Run(sin);
  assert(sin->opcode() == kNegate && sin->operand(0)->opcode() == kSin);
  opt.Run(tan);
  assert(tan->opcode() == kNegate && tan->operand(0)->opcode() == kTan);
  opt.Run(tanh);
  assert(tanh->opcode() == kNegate && tanh->operand(0)->opcode() == kTanh);
}

void inverses_test() {
  Instruction *x = CreateParameter();

  Instruction *exp = CreateUnary(kExp, x);
  Instruction *log = CreateUnary(kLog, x);
  Instruction *neg = CreateUnary(kNegate, x);

  Instruction *log_exp = CreateUnary(kLog, exp);
  Instruction *exp_log = CreateUnary(kExp, log);
  Instruction *neg_neg = CreateUnary(kNegate, neg);

  Optimizer opt;
  opt.Run(log_exp);
  assert(log_exp->opcode() == kParameter);
  opt.Run(exp_log);
  assert(exp_log->opcode() == kParameter);
  opt.Run(neg_neg);
  assert(neg_neg->opcode() == kParameter);
}

void multiply_exp_with_exp_neg_test() {
  Instruction *x = CreateParameter();

  Instruction *exp_x = CreateUnary(kExp, x);
  Instruction *neg = CreateUnary(kNegate, x);
  Instruction *exp_neg = CreateUnary(kExp, neg);
  Instruction *mul = CreateBinary(kMultiply, exp_x, exp_neg);

  Optimizer opt;
  opt.Run(mul);
  assert(IsConstantWithValue(mul, 1));
}

void multiply_powers_test() {
  Instruction *x = CreateParameter();

  Instruction *two = CreateConstant(2);
  Instruction *pow1 = CreateBinary(kPower, x, two);
  Instruction *four = CreateConstant(4);
  Instruction *pow2 = CreateBinary(kPower, x, four);
  Instruction *mul = CreateBinary(kMultiply, pow1, pow2);

  Optimizer opt;
  opt.Run(mul);
  assert(mul->opcode() == kPower && IsConstantWithValue(mul->operand(1), 6));
}

void divide_power_test() {
  Instruction *x = CreateParameter();

  Instruction *three = CreateConstant(3);
  Instruction *pow = CreateBinary(kPower, x, three);
  Instruction *one = CreateConstant(1);
  Instruction *div = CreateBinary(kDivide, one, pow);

  Optimizer opt;
  opt.Run(div);
  assert(div->opcode() == kPower && IsConstantWithValue(div->operand(1), -3));
}

void divide_constant_to_multiply_test() {
  Instruction *x = CreateParameter();

  Instruction *two = CreateConstant(2);
  Instruction *div = CreateBinary(kDivide, x, two);

  Optimizer opt;
  opt.Run(div);
  assert(div->opcode() == kMultiply &&
         IsConstantWithValue(div->operand(0), 0.5));
}

void divide_sin_cos_to_tan_test() {
  Instruction *x = CreateParameter();

  Instruction *sin = CreateUnary(kSin, x);
  Instruction *cos = CreateUnary(kCos, x);
  Instruction *div = CreateBinary(kDivide, sin, cos);

  Optimizer opt;
  opt.Run(div);
  assert(div->opcode() == kTan);
}

void abs_exp_test() {
  Instruction *x = CreateParameter();

  Instruction *exp = CreateUnary(kExp, x);
  Instruction *abs = CreateUnary(kAbs, exp);

  Optimizer opt;
  opt.Run(abs);
  assert(abs->opcode() == kExp);
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

void monotone_test() {
  Instruction *x = CreateParameter();
  Instruction *y = CreateParameter();

  Instruction *exp1 = CreateUnary(kExp, x);
  Instruction *exp2 = CreateUnary(kExp, y);
  Instruction *neg1 = CreateUnary(kNegate, x);
  Instruction *neg2 = CreateUnary(kNegate, y);

  Instruction *max1 = CreateBinary(kMaximum, exp1, exp2);
  Instruction *max2 = CreateBinary(kMaximum, neg1, neg2);
  Instruction *min1 = CreateBinary(kMinimum, exp1, exp2);
  Instruction *min2 = CreateBinary(kMinimum, neg1, neg2);

  Optimizer opt;
  opt.Run(max1);
  assert(max1->opcode() == kExp && max1->operand(0)->opcode() == kMaximum);
  opt.Run(max2);
  assert(max2->opcode() == kNegate && max2->operand(0)->opcode() == kMinimum);
  opt.Run(min1);
  assert(min1->opcode() == kExp && min1->operand(0)->opcode() == kMinimum);
  opt.Run(min2);
  assert(min2->opcode() == kNegate && min2->operand(0)->opcode() == kMaximum);
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

void multiply_power_with_parameter_test() {
  Instruction *x = CreateParameter();

  Instruction *two = CreateConstant(2);
  Instruction *pow = CreateBinary(kPower, x, two);
  Instruction *mul = CreateBinary(kMultiply, x, pow);

  Optimizer opt;
  opt.Run(mul);
  assert(mul->opcode() == kPower && IsConstantWithValue(mul->operand(1), 3));
}

void sqrt_square_test() {
  Instruction *x = CreateParameter();

  Instruction *two = CreateConstant(2);
  Instruction *pow = CreateBinary(kPower, x, two);
  Instruction *sqrt = CreateUnary(kSqrt, pow);

  Optimizer opt;
  opt.Run(sqrt);
  assert(sqrt->opcode() == kAbs);
}

void mul_negates_test() {
  Instruction *x = CreateParameter();
  Instruction *y = CreateParameter();

  Instruction *neg_x = CreateUnary(kNegate, x);
  Instruction *neg_y = CreateUnary(kNegate, y);
  Instruction *mul = CreateBinary(kMultiply, neg_x, neg_y);

  Optimizer opt;
  opt.Run(mul);
  assert(mul->operand(0)->opcode() == kParameter &&
         mul->operand(1)->opcode() == kParameter);
}

void mul_div_test() {
  Instruction *x = CreateParameter();
  Instruction *y = CreateParameter();
  Instruction *z = CreateParameter();

  Instruction *div = CreateBinary(kDivide, y, z);
  Instruction *mul = CreateBinary(kMultiply, x, div);

  Optimizer opt;
  opt.Run(mul);
  assert(mul->opcode() == kDivide);
}

void inv_trigonometric_test() {
  Instruction *x = CreateParameter();

  Instruction *cos_acos = CreateUnary(kCos, CreateUnary(kAcos, x));
  Instruction *sin_asin = CreateUnary(kSin, CreateUnary(kAsin, x));
  Instruction *tan_atan = CreateUnary(kTan, CreateUnary(kAtan, x));

  Optimizer opt;
  opt.Run(cos_acos);
  assert(cos_acos->opcode() == kParameter);
  opt.Run(sin_asin);
  assert(sin_asin->opcode() == kParameter);
  opt.Run(tan_atan);
  assert(tan_atan->opcode() == kParameter);
}

void inv_hyperbolic_test() {
  Instruction *x = CreateParameter();

  Instruction *acosh_cosh = CreateUnary(kAcosh, CreateUnary(kCosh, x));
  Instruction *asinh_sinh = CreateUnary(kAsinh, CreateUnary(kSinh, x));
  Instruction *atanh_tanh = CreateUnary(kAtanh, CreateUnary(kTanh, x));
  Instruction *cosh_acosh = CreateUnary(kCosh, CreateUnary(kAcosh, x));
  Instruction *sinh_asinh = CreateUnary(kSinh, CreateUnary(kAsinh, x));
  Instruction *tanh_atanh = CreateUnary(kTanh, CreateUnary(kAtanh, x));

  Optimizer opt;
  opt.Run(acosh_cosh);
  assert(acosh_cosh->opcode() == kAbs);
  opt.Run(asinh_sinh);
  assert(asinh_sinh->opcode() == kParameter);
  opt.Run(atanh_tanh);
  assert(atanh_tanh->opcode() == kParameter);
  opt.Run(cosh_acosh);
  assert(cosh_acosh->opcode() == kParameter);
  opt.Run(sinh_asinh);
  assert(sinh_asinh->opcode() == kParameter);
  opt.Run(tanh_atanh);
  assert(tanh_atanh->opcode() == kParameter);
}

void homomorphism_test() {
  Instruction *x = CreateParameter();
  Instruction *y = CreateParameter();

  Instruction *add_logs =
      CreateBinary(kAdd, CreateUnary(kLog, x), CreateUnary(kLog, y));
  Instruction *div_abss =
      CreateBinary(kDivide, CreateUnary(kAbs, x), CreateUnary(kAbs, y));
  Instruction *div_exps =
      CreateBinary(kDivide, CreateUnary(kExp, x), CreateUnary(kExp, y));
  Instruction *mul_abss =
      CreateBinary(kMultiply, CreateUnary(kAbs, x), CreateUnary(kAbs, y));
  Instruction *mul_exps =
      CreateBinary(kMultiply, CreateUnary(kExp, x), CreateUnary(kExp, y));
  Instruction *sub_logs =
      CreateBinary(kSubtract, CreateUnary(kLog, x), CreateUnary(kLog, y));

  Optimizer opt;
  opt.Run(add_logs);
  assert(add_logs->opcode() == kLog &&
         add_logs->operand(0)->opcode() == kMultiply);
  opt.Run(div_abss);
  assert(div_abss->opcode() == kAbs &&
         div_abss->operand(0)->opcode() == kDivide);
  opt.Run(div_exps);
  assert(div_exps->opcode() == kExp &&
         div_exps->operand(0)->opcode() == kSubtract);
  opt.Run(mul_abss);
  assert(mul_abss->opcode() == kAbs &&
         mul_abss->operand(0)->opcode() == kMultiply);
  opt.Run(mul_exps);
  assert(mul_exps->opcode() == kExp && mul_exps->operand(0)->opcode() == kAdd);
  opt.Run(sub_logs);
  assert(sub_logs->opcode() == kLog &&
         sub_logs->operand(0)->opcode() == kDivide);
}

int main() {
  unary_constant_folding_test();
  binary_constant_folding_test();
  binary_canonicalization_test();
  even_negate_test();
  odd_negate_test();
  monotone_test();
  inverses_test();
  homomorphism_test();
  inv_trigonometric_test();
  inv_hyperbolic_test();

  multiply_exp_with_exp_neg_test();
  multiply_powers_test();
  divide_power_test();
  divide_constant_to_multiply_test();
  divide_sin_cos_to_tan_test();
  abs_exp_test();

  fold_add_0_test();
  multiply_zero_test();
  multiply_one_test();
  factor_add_multiply_test();
  add_to_multiply_test();
  divide_divide_test();
  multiply_power_common_exponent_test();
  multiply_power_common_base_test();
  negate_subtract_folding_test();
  pow_zero_test();
  pow_one_test();
  zero_pow_test();
  one_pow_test();
  power_power_test();

  // Currently failing tests
  // multiply_power_with_parameter_test();
  // sqrt_square_test();
  // mul_negates_test();
  // mul_div_test();
  return 0;
}
