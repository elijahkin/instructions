#include <cassert>
#include <vector>

#include "opcode.h"

int num_created_instructions = 0;

class Instruction {
private:
  Opcode opcode_;
  std::vector<Instruction *> operands_;
  int id_;

  Instruction(Opcode opcode, const std::vector<Instruction *> &operands) {
    opcode_ = opcode;
    operands_ = operands;
    id_ = num_created_instructions;
    num_created_instructions++;
  }

public:
  Opcode opcode() { return opcode_; }

  std::vector<Instruction *> operands() { return operands_; }

  Instruction *operand(size_t i) { return operands_[i]; }

  std::string to_string() {
    auto opcode_id_string = [](Instruction *instr) {
      return opcode_to_string(instr->opcode_) + '.' +
             std::to_string(instr->id_);
    };

    std::string str;
    for (auto operand : operands()) {
      str += operand->to_string();
    }
    str += '\n';
    str += opcode_id_string(this);
    str += " = ";
    str += opcode_to_string(opcode_);
    str += '(';
    for (size_t i = 0; i < operands_.size(); ++i) {
      if (i != 0) {
        str += ", ";
      }
      str += opcode_id_string(this->operand(i));
    }
    str += ')';
    return str;
  }

  friend Instruction *CreateConstant(double value);

  friend Instruction *CreateParameter();

  friend Instruction *CreateRng();

  friend Instruction *CreateUnary(Opcode opcode, Instruction *operand);

  friend Instruction *CreateBinary(Opcode opcode, Instruction *lhs,
                                   Instruction *rhs);

  friend bool ReplaceInstruction(Instruction *old_instruction,
                                 Instruction *new_instruction);
};

Instruction *CreateParameter() { return new Instruction(kParameter, {}); }

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

bool ReplaceInstruction(Instruction *old_instr, Instruction *new_instr) {
  std::memcpy(old_instr, new_instr, sizeof(*new_instr));
  return true;
}
