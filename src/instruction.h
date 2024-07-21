#include <cassert>
#include <vector>

#include "logging.h"
#include "opcode.h"

int num_created_instructions = 0;

class Instruction {
protected:
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

  friend Instruction *CreateParameter();

  friend Instruction *CreateRng();

  friend Instruction *CreateUnary(Opcode opcode, Instruction *operand);

  friend Instruction *CreateBinary(Opcode opcode, Instruction *lhs,
                                   Instruction *rhs);

  friend bool ReplaceInstruction(Instruction *old_instruction,
                                 Instruction *new_instruction);

  friend bool IsConstantWithValue(Instruction *instruction, double value);
};

Instruction *CreateParameter() { return new Instruction(kParameter, {}); }

Instruction *CreateRng() { return new Instruction(kRng, {}); }

Instruction *CreateUnary(Opcode opcode, Instruction *operand) {
  assert(Arity(opcode) == 1);
  return new Instruction(opcode, {operand});
}

Instruction *CreateBinary(Opcode opcode, Instruction *lhs, Instruction *rhs) {
  assert(Arity(opcode) == 2);
  return new Instruction(opcode, {lhs, rhs});
}

// TODO maybe use std::optional for value instead of creating a new class
class ConstantInstruction : public Instruction {
private:
  double value_;

  ConstantInstruction(double value) : Instruction(kConstant, {}) {
    value_ = value;
  }

public:
  double value() { return value_; }

  friend Instruction *CreateConstant(double value);
};

Instruction *CreateConstant(double value) {
  return new ConstantInstruction(value);
}

bool IsConstantWithValue(Instruction *instruction, double value) {
  return (instruction->opcode() == kConstant) &&
         (static_cast<ConstantInstruction *>(instruction)->value() == value);
}

bool ReplaceInstruction(Instruction *old_instr, Instruction *new_instr) {
  int num_bytes = sizeof(Instruction);
  if (new_instr->opcode() == kConstant) {
    num_bytes = sizeof(ConstantInstruction);
  }
  std::memcpy(old_instr, new_instr, num_bytes);
  return true;
}
