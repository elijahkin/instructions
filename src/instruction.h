#include <cassert>
#include <vector>

#include "opcode.h"

int num_created_instructions = 0;

class Instruction {
private:
  Opcode opcode_;
  std::vector<Instruction *> operands_;
  std::vector<Instruction *> users_;
  int id_;

  Instruction(Opcode opcode, const std::vector<Instruction *> &operands) {
    opcode_ = opcode;
    operands_ = operands;
    id_ = num_created_instructions;

    // add this instruction as a user of its operands
    for (auto operand : operands_) {
      operand->users_.push_back(this);
    }

    num_created_instructions++;
  }

public:
  Opcode opcode() { return opcode_; }

  std::vector<Instruction *> operands() { return operands_; }

  Instruction *operand(size_t i) { return operands_[i]; }

  std::vector<Instruction *> users() { return users_; }

  std::string to_string() {
    std::string str;
    for (auto operand : operands()) {
      str += operand->to_string();
      str += '\n';
    }
    str += opcode_to_string(opcode_);
    str += '.';
    str += std::to_string(id_);
    str += " = ";
    str += opcode_to_string(opcode_);
    str += '(';
    for (size_t i = 0; i < operands_.size(); ++i) {
      if (i != 0) {
        str += ", ";
      }
      str += opcode_to_string(operands_[i]->opcode());
      str += '.';
      str += std::to_string(operands_[i]->id_);
    }
    str += ')';
    return str;
  }

  int arity() { return operands_.size(); }

  friend Instruction *CreateVariable();

  friend Instruction *CreateConstant(double value);

  friend Instruction *CreateRng();

  friend Instruction *CreateUnary(Opcode opcode, Instruction *operand);

  friend Instruction *CreateBinary(Opcode opcode, Instruction *lhs,
                                   Instruction *rhs);

  friend bool ReplaceInstruction(Instruction *old_instruction,
                                 Instruction *new_instruction);
};

Instruction *CreateVariable() { return new Instruction(kVariable, {}); }

Instruction *CreateConstant(double value) {
  return new Instruction(kConstant, {});
}

Instruction *CreateRng() { return new Instruction(kRng, {}); }

Instruction *CreateUnary(Opcode opcode, Instruction *operand) {
  assert(Arity(opcode) == 1);
  return new Instruction(opcode, {operand});
}

Instruction *CreateBinary(Opcode opcode, Instruction *lhs, Instruction *rhs) {
  assert(Arity(opcode) == 2);
  return new Instruction(opcode, {lhs, rhs});
}
