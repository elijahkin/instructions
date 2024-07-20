#include <iostream>

#include "instruction.h"

bool ReplaceInstruction(Instruction *old_instruction,
                        Instruction *new_instruction) {
  for (auto user : old_instruction->users()) {
    for (int i = 0; i < old_instruction->arity(); ++i) {
      if (user->operands_[i] == old_instruction) {
        user->operands_[i] = new_instruction;
      }
    }
  }
  return true;
}

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
        std::cout << "[OPTIMIZER] xy+xz --> x(y+z)" << std::endl;
        return ReplaceInstruction(
            add, CreateBinary(kMultiply, add->operand(0)->operand(0),
                              CreateBinary(kAdd, add->operand(0)->operand(1),
                                           add->operand(1)->operand(1))));
      }
      if (add->operand(0)->operand(1) == add->operand(1)->operand(1)) {
        std::cout << "[OPTIMIZER] xz+yz --> (x+y)z" << std::endl;
        return ReplaceInstruction(
            add, CreateBinary(kMultiply,
                              CreateBinary(kAdd, add->operand(0)->operand(0),
                                           add->operand(1)->operand(0)),
                              add->operand(0)->operand(1)));
      }
    }

    if (add->operand(0) == add->operand(1)) {
      std::cout << "[OPTIMIZER] x+x --> 2*x" << std::endl;
      return ReplaceInstruction(add, CreateBinary(kMultiply, add->operand(0),
                                                  CreateNullary(kConstant)));
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
      std::cout << "[OPTIMIZER] exp(log(x)) --> x" << std::endl;
      return ReplaceInstruction(exp, exp->operand(0)->operand(0));
    }
    return false;
  }

  bool OptimizeLog(Instruction *log) {
    assert(log->opcode() == kLog);

    if (log->operand(0)->opcode() == kExp) {
      std::cout << "[OPTIMIZER] log(exp(x)) --> x" << std::endl;
      return ReplaceInstruction(log, log->operand(0)->operand(0));
    }
    return false;
  }

  bool OptimizeMultiply(Instruction *multiply) {
    assert(multiply->opcode() == kMultiply);
    // TODO 1*x --> x

    if (multiply->operand(0)->opcode() == kExp &&
        multiply->operand(1)->opcode() == kExp) {
      std::cout << "[OPTIMIZER] exp(x)*exp(y) --> exp(x+y)" << std::endl;
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
      std::cout << "[OPTIMIZER] -(-x) --> x" << std::endl;
      return ReplaceInstruction(negate, negate->operand(0)->operand(0));
    }

    if (negate->operand(0)->opcode() == kSubtract) {
      std::cout << "[OPTIMIZER] -(x-y) --> y-x" << std::endl;
      return ReplaceInstruction(
          negate, CreateBinary(kSubtract, negate->operand(0)->operand(1),
                               negate->operand(0)->operand(0)));
    }
    return false;
  }
};
