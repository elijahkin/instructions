#include <cassert>
#include <vector>

#include "opcode.h"

class Instruction {
private:
  Opcode opcode_;
  std::vector<Instruction *> operands_;
  std::vector<Instruction *> users_;

  Instruction(Opcode opcode, const std::vector<Instruction *> &operands) {
    opcode_ = opcode;
    operands_ = operands;
    // users_ = {};
    // add this instruction as a user of its operands
    for (auto operand : operands_) {
      operand->users_.push_back(this);
    }
  }

public:
  Opcode opcode() { return opcode_; }

  std::vector<Instruction *> operands() { return operands_; }

  Instruction *operand(size_t i) { return operands_[i]; }

  std::vector<Instruction *> users() { return users_; }

  std::string to_string() {
    std::string str = opcode_to_string(opcode_);
    str += '(';
    for (size_t i = 0; i < operands_.size(); ++i) {
      if (i != 0) {
        str += ", ";
      }
      str += operands_[i]->to_string();
    }
    str += ')';
    return str;
  }

  int arity() { return operands_.size(); }

  friend Instruction *CreateNullary(Opcode opcode);

  friend Instruction *CreateUnary(Opcode opcode, Instruction *operand);

  friend Instruction *CreateBinary(Opcode opcode, Instruction *lhs,
                                   Instruction *rhs);

  friend bool ReplaceInstruction(Instruction *old_instruction,
                                 Instruction *new_instruction);
};

Instruction *CreateNullary(Opcode opcode) {
  assert(Arity(opcode) == 0);
  return new Instruction(opcode, {});
}

Instruction *CreateUnary(Opcode opcode, Instruction *operand) {
  assert(Arity(opcode) == 1);
  return new Instruction(opcode, {operand});
}

Instruction *CreateBinary(Opcode opcode, Instruction *lhs, Instruction *rhs) {
  assert(Arity(opcode) == 2);
  return new Instruction(opcode, {lhs, rhs});
}
