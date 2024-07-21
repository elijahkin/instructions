#include "instruction.h"
#include "logging.h"

class Optimizer {
public:
  bool Optimize(Instruction *instruction) {
    bool changed = false;

    switch (instruction->opcode()) {
    case kAbs:
      changed = OptimizeAbs(instruction);
      break;
    case kAdd:
      changed = OptimizeAdd(instruction);
      break;
    case kConstant:
      changed = OptimizeConstant(instruction);
      break;
    case kDivide:
      changed = OptimizeDivide(instruction);
      break;
    case kExp:
      changed = OptimizeExp(instruction);
      break;
    case kLog:
      changed = OptimizeLog(instruction);
      break;
    case kMaximum:
      changed = OptimizeMaximum(instruction);
      break;
    case kMinimum:
      changed = OptimizeMinimum(instruction);
      break;
    case kMultiply:
      changed = OptimizeMultiply(instruction);
      break;
    case kNegate:
      changed = OptimizeNegate(instruction);
      break;
    case kPower:
      changed = OptimizePower(instruction);
      break;
    case kSubtract:
      changed = OptimizeSubtract(instruction);
      break;
    default:
      break;
    }

    for (auto operand : instruction->operands()) {
      changed = Optimize(operand);
    }

    if (changed) {
      Optimize(instruction);
    }
    return changed;
  }

  bool OptimizeAbs(Instruction *abs) {
    assert(abs->opcode() == kAbs);

    Instruction *operand = abs->operand(0);

    if (operand->opcode() == kNegate) {
      VLOG(10) << "|-x| --> |x|";
      return ReplaceInstruction(abs, CreateUnary(kAbs, operand->operand(0)));
    }
    return false;
  }

  bool OptimizeAdd(Instruction *add) {
    assert(add->opcode() == kAdd);

    Instruction *lhs = add->operand(0);
    Instruction *rhs = add->operand(1);

    // TODO x+0 --> x

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

  bool OptimizeConstant(Instruction *constant) {
    assert(constant->opcode() == kConstant);

    // TODO if all operands are constants, fold
    return false;
  }

  bool OptimizeDivide(Instruction *divide) {
    assert(divide->opcode() == kDivide);

    Instruction *lhs = divide->operand(0);
    Instruction *rhs = divide->operand(1);

    // TODO x / constant --> x * constant

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

  bool OptimizeExp(Instruction *exp) {
    assert(exp->opcode() == kExp);

    Instruction *operand = exp->operand(0);

    if (operand->opcode() == kLog) {
      VLOG(10) << "exp(log(x)) --> x";
      return ReplaceInstruction(exp, operand->operand(0));
    }
    return false;
  }

  bool OptimizeLog(Instruction *log) {
    assert(log->opcode() == kLog);

    Instruction *operand = log->operand(0);

    if (operand->opcode() == kExp) {
      VLOG(10) << "log(exp(x)) --> x";
      return ReplaceInstruction(log, operand->operand(0));
    }
    return false;
  }

  bool OptimizeMaximum(Instruction *maximum) {
    assert(maximum->opcode() == kMaximum);

    // TODO max(f(x), f(y)) --> f(max(x, y))
    // where f is monotonically increasing

    // TODO max(f(x), f(y)) --> f(min(x, y))
    // where f is monotonically decreasing
    return false;
  }

  bool OptimizeMinimum(Instruction *minimum) {
    assert(minimum->opcode() == kMinimum);

    // TODO monotonic rewrites as with maximum
    return false;
  }

  bool OptimizeMultiply(Instruction *multiply) {
    assert(multiply->opcode() == kMultiply);

    Instruction *lhs = multiply->operand(0);
    Instruction *rhs = multiply->operand(1);

    // TODO 0*x --> x

    // TODO 1*x --> x

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

  bool OptimizeNegate(Instruction *negate) {
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

  bool OptimizePower(Instruction *power) {
    assert(power->opcode() == kPower);

    Instruction *lhs = power->operand(0);
    Instruction *rhs = power->operand(1);

    // TODO pow(x,0) --> 1

    // TODO pow(x,1) --> x

    // TODO pow(x,2) --> x*x

    // TODO pow(0,x) --> 0

    // TODO pow(1,x) --> 1

    if (lhs->opcode() == kPower) {
      VLOG(10) << "pow(pow(x,y),z) --> pow(x,y*z)";
      return ReplaceInstruction(
          power, CreateBinary(kPower, lhs->operand(0),
                              CreateBinary(kMultiply, lhs->operand(1), rhs)));
    }
    return false;
  }

  bool OptimizeSubtract(Instruction *subtract) {
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
