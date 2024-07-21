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

    if (abs->operand(0)->opcode() == kNegate) {
      VLOG(10) << "|-x| --> |x|";
      return ReplaceInstruction(abs,
                                CreateUnary(kAbs, abs->operand(0)->operand(0)));
    }
    return false;
  }

  bool OptimizeAdd(Instruction *add) {
    assert(add->opcode() == kAdd);

    if (add->operand(0)->opcode() == kMultiply &&
        add->operand(1)->opcode() == kMultiply) {
      if (add->operand(0)->operand(0) == add->operand(1)->operand(0)) {
        VLOG(10) << "xy+xz --> x(y+z)";
        return ReplaceInstruction(
            add, CreateBinary(kMultiply, add->operand(0)->operand(0),
                              CreateBinary(kAdd, add->operand(0)->operand(1),
                                           add->operand(1)->operand(1))));
      }
      if (add->operand(0)->operand(1) == add->operand(1)->operand(1)) {
        VLOG(10) << "xz+yz --> (x+y)z";
        return ReplaceInstruction(
            add, CreateBinary(kMultiply,
                              CreateBinary(kAdd, add->operand(0)->operand(0),
                                           add->operand(1)->operand(0)),
                              add->operand(0)->operand(1)));
      }
    }

    if (add->operand(0) == add->operand(1)) {
      VLOG(10) << "x+x --> 2*x";
      return ReplaceInstruction(
          add, CreateBinary(kMultiply, add->operand(0), CreateConstant(2)));
    }

    // TODO x+0 --> x
    return false;
  }

  bool OptimizeConstant(Instruction *constant) {
    assert(constant->opcode() == kConstant);

    // TODO if all operands are constants, fold
    return false;
  }

  bool OptimizeDivide(Instruction *divide) {
    assert(divide->opcode() == kDivide);

    // TODO x / constant --> x * constant

    if (divide->operand(0)->opcode() == kDivide) {
      VLOG(10) << "(x/y)/z --> x/(y*z)";
      return ReplaceInstruction(
          divide,
          CreateBinary(kDivide, divide->operand(0)->operand(0),
                       CreateBinary(kMultiply, divide->operand(0)->operand(1),
                                    divide->operand(1))));
    }
    if (divide->operand(1)->opcode() == kDivide) {
      VLOG(10) << "x/(y/z) --> (x*z)/y";
      return ReplaceInstruction(
          divide, CreateBinary(kDivide,
                               CreateBinary(kMultiply, divide->operand(0),
                                            divide->operand(1)->operand(1)),
                               divide->operand(1)->operand(0)));
    }
    return false;
  }

  bool OptimizeExp(Instruction *exp) {
    assert(exp->opcode() == kExp);

    if (exp->operand(0)->opcode() == kLog) {
      VLOG(10) << "exp(log(x)) --> x";
      return ReplaceInstruction(exp, exp->operand(0)->operand(0));
    }
    return false;
  }

  bool OptimizeLog(Instruction *log) {
    assert(log->opcode() == kLog);

    if (log->operand(0)->opcode() == kExp) {
      VLOG(10) << "log(exp(x)) --> x";
      return ReplaceInstruction(log, log->operand(0)->operand(0));
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

    // TODO 1*x --> x

    if (multiply->operand(0)->opcode() == kExp &&
        multiply->operand(1)->opcode() == kExp) {
      VLOG(10) << "exp(x)*exp(y) --> exp(x+y)";
      return ReplaceInstruction(
          multiply,
          CreateUnary(kExp, CreateBinary(kAdd, multiply->operand(0)->operand(0),
                                         multiply->operand(1)->operand(0))));
    }
    return false;
  }

  bool OptimizeNegate(Instruction *negate) {
    assert(negate->opcode() == kNegate);

    if (negate->operand(0)->opcode() == kNegate) {
      VLOG(10) << "-(-x) --> x";
      return ReplaceInstruction(negate, negate->operand(0)->operand(0));
    }

    if (negate->operand(0)->opcode() == kSubtract) {
      VLOG(10) << "-(x-y) --> y-x";
      return ReplaceInstruction(
          negate, CreateBinary(kSubtract, negate->operand(0)->operand(1),
                               negate->operand(0)->operand(0)));
    }
    return false;
  }

  bool OptimizePower(Instruction *power) {
    assert(power->opcode() == kPower);

    // TODO pow(x,0) --> 1

    // TODO pow(x,1) --> x

    // TODO pow(x,2) --> x*x

    // TODO pow(0,x) --> 0

    // TODO pow(1,x) --> 1

    // TODO pow(x,y)*pow(x,z) --> pow(x,y+z)

    // TODO pow(pow(x,y),z) --> pow(x,y*z)

    // TODO pow(x,z)*pow(y,z) --> pow(x*y,z)
    return false;
  }

  bool OptimizeSubtract(Instruction *subtract) {
    assert(subtract->opcode() == kSubtract);

    if (subtract->operand(0)->opcode() == kLog &&
        subtract->operand(0)->opcode() == kLog) {
      VLOG(10) << "log(x)-log(y) --> log(x/y)";
      return ReplaceInstruction(
          subtract,
          CreateUnary(kLog,
                      CreateBinary(kDivide, subtract->operand(0)->operand(0),
                                   subtract->operand(1)->operand(0))));
    }
    return false;
  }
};
