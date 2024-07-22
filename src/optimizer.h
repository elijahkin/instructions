#include <cmath>

#include "instruction.h"

class Optimizer {
public:
  bool Run(Instruction *instruction) {
    bool changed = false;

    switch (Arity(instruction->opcode())) {
    case 1:
      changed |= HandleUnary(instruction);
      break;
    case 2:
      changed |= HandleBinary(instruction);
      break;
    }

    switch (instruction->opcode()) {
    case kAbs:
      changed |= HandleAbs(instruction);
      break;
    case kAdd:
      changed |= HandleAdd(instruction);
      break;
    case kDivide:
      changed |= HandleDivide(instruction);
      break;
    case kExp:
      changed |= HandleExp(instruction);
      break;
    case kLog:
      changed |= HandleLog(instruction);
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

    for (auto operand : instruction->operands()) {
      changed = Run(operand);
    }

    if (changed) {
      Run(instruction);
    }
    return changed;
  }

  bool HandleUnary(Instruction *unary) {
    Instruction *operand = unary->operand(0);

    // TODO Unify with analogous rewrite in HandleBinary
    if (AllConstantOperands(unary)) {
      VLOG(10) << "Folding unary operation with constant operand";
      return ReplaceInstruction(unary, CreateConstant(Evaluate(unary)));
    }

    // Inverse op cancellation
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
    bool commutative =
        (binary->opcode() == kAdd) | (binary->opcode() == kMultiply);
    if (commutative && rhs->opcode() == kConstant) {
      VLOG(10) << "Canonicalizing constant to lhs of commutative binary op";
      return ReplaceInstruction(binary,
                                CreateBinary(binary->opcode(), rhs, lhs));
    }

    // TODO eliminate left/right identities and annihilators

    // TODO homomorphisms
    return false;
  }

  bool HandleAbs(Instruction *abs) {
    assert(abs->opcode() == kAbs);

    Instruction *operand = abs->operand(0);

    if (operand->opcode() == kNegate) {
      VLOG(10) << "|-x| --> |x|";
      return ReplaceInstruction(abs, CreateUnary(kAbs, operand->operand(0)));
    }
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
    return false;
  }

  bool HandleDivide(Instruction *divide) {
    assert(divide->opcode() == kDivide);

    Instruction *lhs = divide->operand(0);
    Instruction *rhs = divide->operand(1);

    // if (rhs->opcode() == kConstant) {
    //   VLOG(10) << "x/c --> x*c";
    //   return TODO;
    // }

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

  bool HandleExp(Instruction *exp) {
    assert(exp->opcode() == kExp);

    Instruction *operand = exp->operand(0);

    if (operand->opcode() == kLog) {
      VLOG(10) << "exp(log(x)) --> x";
      return ReplaceInstruction(exp, operand->operand(0));
    }
    return false;
  }

  bool HandleLog(Instruction *log) {
    assert(log->opcode() == kLog);

    Instruction *operand = log->operand(0);

    if (operand->opcode() == kExp) {
      VLOG(10) << "log(exp(x)) --> x";
      return ReplaceInstruction(log, operand->operand(0));
    }
    return false;
  }

  bool HandleMaximum(Instruction *maximum) {
    assert(maximum->opcode() == kMaximum);

    // TODO max(f(x), f(y)) --> f(max(x, y))
    // where f is monotonically increasing

    // TODO max(f(x), f(y)) --> f(min(x, y))
    // where f is monotonically decreasing
    return false;
  }

  bool HandleMinimum(Instruction *minimum) {
    assert(minimum->opcode() == kMinimum);

    // TODO monotonic rewrites as with maximum
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

    // TODO This can be combined with the similar identity element rewrite for
    // add
    if (IsConstantWithValue(lhs, 1)) {
      VLOG(10) << "1*x --> x";
      return ReplaceInstruction(multiply, rhs);
    }

    // TODO The following two rewrites can be unified into a single rewrite
    // targeting homomorphisms
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

    if (operand->opcode() == kNegate) {
      VLOG(10) << "-(-x) --> x";
      return ReplaceInstruction(negate, operand->operand(0));
    }

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

    if (lhs->opcode() == kLog && rhs->opcode() == kLog) {
      VLOG(10) << "log(x)-log(y) --> log(x/y)";
      return ReplaceInstruction(
          subtract, CreateUnary(kLog, CreateBinary(kDivide, lhs->operand(0),
                                                   rhs->operand(0))));
    }
    return false;
  }
};
