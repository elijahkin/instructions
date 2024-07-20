#include "instruction.h"
#include "logging.h"

class Optimizer {
public:
  void Optimize(Instruction *instruction) {
    bool changed = false;

    switch (instruction->opcode()) {
    case kAdd:
      changed = OptimizeAdd(instruction);
      break;
    case kConstant:
      changed = OptimizeConstant(instruction);
      break;
    case kExp:
      changed = OptimizeExp(instruction);
      break;
    case kLog:
      changed = OptimizeLog(instruction);
      break;
    case kMultiply:
      changed = OptimizeMultiply(instruction);
      break;
    case kNegate:
      changed = OptimizeNegate(instruction);
      break;
    default:
      break;
    }
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
};
