#include <cmath>

#include "instruction.h"

class Optimizer {
public:
  bool Run(Instruction *instruction) {
    bool changed = false;

    // Recurse on the instruction's operands before optimizing this one
    for (auto operand : instruction->operands()) {
      changed = Run(operand);
    }

    switch (Arity(instruction->opcode())) {
    case 1:
      changed |= HandleUnary(instruction);
      break;
    case 2:
      changed |= HandleBinary(instruction);
      break;
    }

    switch (instruction->opcode()) {
    case kAdd:
      changed |= HandleAdd(instruction);
      break;
    case kDivide:
      changed |= HandleDivide(instruction);
      break;
    case kMaximum:
      changed |= HandleMaximum(instruction);
      break;
    case kMinimum:
      changed |= HandleMinimum(instruction);
      break;
    case kMultiply:
      changed |= HandleMultiply(instruction);
      break;
    case kNegate:
      changed |= HandleNegate(instruction);
      break;
    case kPower:
      changed |= HandlePower(instruction);
      break;
    case kSubtract:
      changed |= HandleSubtract(instruction);
      break;
    default:
      break;
    }

    if (changed) {
      Run(instruction);
    }
    return changed;
  }

  bool HandleUnary(Instruction *unary) {
    Instruction *operand = unary->operand(0);

    // TODO unify with analogous rewrite in HandleBinary
    if (AllConstantOperands(unary)) {
      VLOG(10) << "Folding unary operation with constant operand";
      return ReplaceInstruction(unary, CreateConstant(Evaluate(unary)));
    }

    if (auto inv_op = Inverse(unary->opcode()); inv_op.has_value()) {
      if (operand->opcode() == Inverse(unary->opcode()).value()) {
        VLOG(10) << "f(g(x)) --> x where f and g are inverses";
        return ReplaceInstruction(unary, operand->operand(0));
      }
    }

    if (operand->opcode() == kNegate && IsEven(unary->opcode())) {
      VLOG(10) << "f(-x) --> f(x) where f is even";
      return ReplaceInstruction(
          unary, CreateUnary(unary->opcode(), operand->operand(0)));
    }

    if (operand->opcode() == kNegate && IsOdd(unary->opcode())) {
      VLOG(10) << "f(-x) --> -f(x) where f is odd";
      return ReplaceInstruction(
          unary, CreateUnary(kNegate, CreateUnary(unary->opcode(),
                                                  operand->operand(0))));
    }
    return false;
  }

  bool HandleBinary(Instruction *binary) {
    Instruction *lhs = binary->operand(0);
    Instruction *rhs = binary->operand(1);

    // Fold operations whose every operand is constant
    if (AllConstantOperands(binary)) {
      VLOG(10) << "Folding binary operation with all constant operands";
      return ReplaceInstruction(binary, CreateConstant(Evaluate(binary)));
    }

    // Canonicalize constants to be on the lhs for commutative ops
    if (IsCommutative(binary->opcode()) && rhs->opcode() == kConstant) {
      VLOG(10) << "Canonicalizing constant to lhs of commutative binary op";
      return ReplaceInstruction(binary,
                                CreateBinary(binary->opcode(), rhs, lhs));
    }

    // TODO Fold expressions equal to 0 (0*x, pow(0,x), x-x)

    // TODO Fold expressions equal to 1

    // TODO Fold identity elements

    // TODO Strength reduction

    // TODO Homomorphisms
    return false;
  }

  bool HandleAdd(Instruction *add) {
    assert(add->opcode() == kAdd);

    Instruction *lhs = add->operand(0);
    Instruction *rhs = add->operand(1);

    if (IsConstantWithValue(lhs, 0)) {
      VLOG(10) << "x+0 --> x";
      return ReplaceInstruction(add, rhs);
    }

    // TODO Maybe canonicalize so we don't need both of these?
    if (rhs->opcode() == kNegate) {
      VLOG(10) << "x+(-y) --> x-y";
      return ReplaceInstruction(add,
                                CreateBinary(kSubtract, lhs, rhs->operand(0)));
    }

    if (lhs->opcode() == kNegate) {
      VLOG(10) << "(-x)+y --> y-x";
      return ReplaceInstruction(add,
                                CreateBinary(kSubtract, rhs, lhs->operand(0)));
    }

    if (lhs->opcode() == kMultiply && rhs->opcode() == kMultiply) {
      if (lhs->operand(0) == rhs->operand(0)) {
        VLOG(10) << "xy+xz --> x(y+z)";
        return ReplaceInstruction(
            add,
            CreateBinary(kMultiply, lhs->operand(0),
                         CreateBinary(kAdd, lhs->operand(1), rhs->operand(1))));
      }
      if (lhs->operand(1) == rhs->operand(1)) {
        VLOG(10) << "xz+yz --> (x+y)z";
        return ReplaceInstruction(
            add,
            CreateBinary(kMultiply,
                         CreateBinary(kAdd, lhs->operand(0), rhs->operand(0)),
                         lhs->operand(1)));
      }
    }

    if (lhs == rhs) {
      VLOG(10) << "x+x --> 2*x";
      return ReplaceInstruction(
          add, CreateBinary(kMultiply, lhs, CreateConstant(2)));
    }

    // TODO pow(sin(x),2)+pow(cos(x),2) --> 1
    return false;
  }

  bool HandleDivide(Instruction *divide) {
    assert(divide->opcode() == kDivide);

    Instruction *lhs = divide->operand(0);
    Instruction *rhs = divide->operand(1);

    if (rhs->opcode() == kConstant) {
      VLOG(10) << "x/c --> x*c";
      return ReplaceInstruction(
          divide,
          CreateBinary(kMultiply, lhs, CreateConstant(1 / Evaluate(rhs))));
    }

    if (lhs == rhs) {
      VLOG(10) << "x/x --> 1";
      return ReplaceInstruction(divide, CreateConstant(1));
    }

    if (lhs->opcode() == kSin && rhs->opcode() == kCos &&
        lhs->operand(0) == rhs->operand(0)) {
      VLOG(10) << "sin(x)/cos(x) --> tan(x)";
      return ReplaceInstruction(divide, CreateUnary(kTan, lhs->operand(0)));
    }

    if (rhs->opcode() == kPower) {
      VLOG(10) << "x/pow(y,z) --> x*pow(y,-z)";
      return ReplaceInstruction(
          divide,
          CreateBinary(kMultiply, lhs,
                       CreateBinary(kPower, rhs->operand(0),
                                    CreateUnary(kNegate, rhs->operand(1)))));
    }

    if (lhs->opcode() == kDivide) {
      VLOG(10) << "(x/y)/z --> x/(y*z)";
      return ReplaceInstruction(
          divide, CreateBinary(kDivide, lhs->operand(0),
                               CreateBinary(kMultiply, lhs->operand(1), rhs)));
    }

    if (rhs->opcode() == kDivide) {
      VLOG(10) << "x/(y/z) --> (x*z)/y";
      return ReplaceInstruction(
          divide,
          CreateBinary(kDivide, CreateBinary(kMultiply, lhs, rhs->operand(1)),
                       rhs->operand(0)));
    }
    return false;
  }

  bool HandleMaximum(Instruction *maximum) {
    assert(maximum->opcode() == kMaximum);

    Instruction *lhs = maximum->operand(0);
    Instruction *rhs = maximum->operand(1);

    // TODO unify monotone rewrites (with sublinear condition?)
    if (lhs->opcode() == rhs->opcode() && IsIncreasing(lhs->opcode())) {
      VLOG(10) << "max(f(x), f(y)) --> f(max(x, y)) where f is increasing";
      return ReplaceInstruction(
          maximum,
          CreateUnary(lhs->opcode(), CreateBinary(kMaximum, lhs->operand(0),
                                                  rhs->operand(0))));
    }

    if (lhs->opcode() == rhs->opcode() && IsDecreasing(lhs->opcode())) {
      VLOG(10) << "max(f(x), f(y)) --> f(min(x, y)) where f is decreasing";
      return ReplaceInstruction(
          maximum,
          CreateUnary(lhs->opcode(), CreateBinary(kMinimum, lhs->operand(0),
                                                  rhs->operand(0))));
    }
    return false;
  }

  bool HandleMinimum(Instruction *minimum) {
    assert(minimum->opcode() == kMinimum);

    Instruction *lhs = minimum->operand(0);
    Instruction *rhs = minimum->operand(1);

    if (lhs->opcode() == rhs->opcode() && IsIncreasing(lhs->opcode())) {
      VLOG(10) << "min(f(x), f(y)) --> f(min(x, y)) where f is increasing";
      return ReplaceInstruction(
          minimum,
          CreateUnary(lhs->opcode(), CreateBinary(kMinimum, lhs->operand(0),
                                                  rhs->operand(0))));
    }

    if (lhs->opcode() == rhs->opcode() && IsDecreasing(lhs->opcode())) {
      VLOG(10) << "min(f(x), f(y)) --> f(max(x, y)) where f is decreasing";
      return ReplaceInstruction(
          minimum,
          CreateUnary(lhs->opcode(), CreateBinary(kMaximum, lhs->operand(0),
                                                  rhs->operand(0))));
    }
    return false;
  }

  bool HandleMultiply(Instruction *multiply) {
    assert(multiply->opcode() == kMultiply);

    Instruction *lhs = multiply->operand(0);
    Instruction *rhs = multiply->operand(1);

    if (IsConstantWithValue(lhs, 0)) {
      VLOG(10) << "0*x --> 0";
      return ReplaceInstruction(multiply, lhs);
    }

    // TODO unify with 0+x --> x add identity rewrite
    if (IsConstantWithValue(lhs, 1)) {
      VLOG(10) << "1*x --> x";
      return ReplaceInstruction(multiply, rhs);
    }

    // TODO unify into a single rewrite targeting homomorphisms
    if (lhs->opcode() == kAbs && rhs->opcode() == kAbs) {
      VLOG(10) << "|x|*|y| --> |x*y|";
      return ReplaceInstruction(
          multiply, CreateUnary(kAbs, CreateBinary(kMultiply, lhs->operand(0),
                                                   rhs->operand(0))));
    }

    if (lhs->opcode() == kExp && rhs->opcode() == kExp) {
      VLOG(10) << "exp(x)*exp(y) --> exp(x+y)";
      return ReplaceInstruction(
          multiply, CreateUnary(kExp, CreateBinary(kAdd, lhs->operand(0),
                                                   rhs->operand(0))));
    }

    if (lhs->opcode() == kPower && rhs->opcode() == kPower) {
      if (lhs->operand(1) == rhs->operand(1)) {
        VLOG(10) << "pow(x,z)*pow(y,z) --> pow(x*y,z)";
        return ReplaceInstruction(
            multiply, CreateBinary(kPower,
                                   CreateBinary(kMultiply, lhs->operand(0),
                                                rhs->operand(0)),
                                   lhs->operand(1)));
      }
      if (lhs->operand(0) == rhs->operand(0)) {
        VLOG(10) << "pow(x,y)*pow(x,z) --> pow(x,y+z)";
        return ReplaceInstruction(
            multiply,
            CreateBinary(kPower, lhs->operand(0),
                         CreateBinary(kAdd, lhs->operand(1), rhs->operand(1))));
      }
    }
    return false;
  }

  bool HandleNegate(Instruction *negate) {
    assert(negate->opcode() == kNegate);

    Instruction *operand = negate->operand(0);

    if (operand->opcode() == kSubtract) {
      VLOG(10) << "-(x-y) --> y-x";
      return ReplaceInstruction(
          negate,
          CreateBinary(kSubtract, operand->operand(1), operand->operand(0)));
    }
    return false;
  }

  bool HandlePower(Instruction *power) {
    assert(power->opcode() == kPower);

    Instruction *lhs = power->operand(0);
    Instruction *rhs = power->operand(1);

    if (IsConstantWithValue(rhs, 0)) {
      VLOG(10) << "pow(x,0) --> 1";
      return ReplaceInstruction(power, CreateConstant(1));
    }

    if (IsConstantWithValue(rhs, 1)) {
      VLOG(10) << "pow(x,1) --> x";
      return ReplaceInstruction(power, lhs);
    }

    if (IsConstantWithValue(rhs, 2)) {
      VLOG(10) << "pow(x,2) --> x*x";
      return ReplaceInstruction(power, CreateBinary(kMultiply, lhs, lhs));
    }

    if (IsConstantWithValue(lhs, 0)) {
      VLOG(10) << "pow(0,x) --> 0";
      return ReplaceInstruction(power, lhs);
    }

    if (IsConstantWithValue(lhs, 1)) {
      VLOG(10) << "pow(1,x) --> 1";
      return ReplaceInstruction(power, lhs);
    }

    if (lhs->opcode() == kPower) {
      VLOG(10) << "pow(pow(x,y),z) --> pow(x,y*z)";
      return ReplaceInstruction(
          power, CreateBinary(kPower, lhs->operand(0),
                              CreateBinary(kMultiply, lhs->operand(1), rhs)));
    }
    return false;
  }

  bool HandleSubtract(Instruction *subtract) {
    assert(subtract->opcode() == kSubtract);

    Instruction *lhs = subtract->operand(0);
    Instruction *rhs = subtract->operand(1);

    if (lhs == rhs) {
      VLOG(10) << "x-x --> 0";
      return ReplaceInstruction(subtract, CreateConstant(0));
    }

    if (lhs->opcode() == kLog && rhs->opcode() == kLog) {
      VLOG(10) << "log(x)-log(y) --> log(x/y)";
      return ReplaceInstruction(
          subtract, CreateUnary(kLog, CreateBinary(kDivide, lhs->operand(0),
                                                   rhs->operand(0))));
    }

    // TODO pow(x,2)-pow(y,2) --> (x-y)*(x+y)
    return false;
  }
};
