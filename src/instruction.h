#include <cassert>
#include <cmath>
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

bool AllConstantOperands(Instruction *instruction) {
  bool all_constant = true;
  for (auto operand : instruction->operands()) {
    all_constant &= (operand->opcode() == kConstant);
  }
  return all_constant;
}

bool ReplaceInstruction(Instruction *old_instr, Instruction *new_instr) {
  int num_bytes = sizeof(Instruction);
  if (new_instr->opcode() == kConstant) {
    num_bytes = sizeof(ConstantInstruction);
  }
  std::memcpy(old_instr, new_instr, num_bytes);
  return true;
}

double Evaluate(Instruction *instruction) {
  std::vector<double> evaluated;
  for (auto operand : instruction->operands()) {
    evaluated.push_back(Evaluate(operand));
  }

  switch (instruction->opcode()) {
  case kAbs:
    return abs(evaluated[0]);
  case kAdd:
    return evaluated[0] + evaluated[1];
  case kAtan2:
    return atan2(evaluated[0], evaluated[0]);
  case kConstant:
    return static_cast<ConstantInstruction *>(instruction)->value();
  case kCos:
    return cos(evaluated[0]);
  case kDivide:
    return evaluated[0] / evaluated[1];
  case kExp:
    return exp(evaluated[0]);
  case kLog:
    return log(evaluated[0]);
  case kMaximum:
    return fmax(evaluated[0], evaluated[1]);
  case kMinimum:
    return fmin(evaluated[0], evaluated[1]);
  case kMultiply:
    return evaluated[0] * evaluated[1];
  case kNegate:
    return -evaluated[0];
  case kPower:
    return pow(evaluated[0], evaluated[1]);
  case kSin:
    return sin(evaluated[0]);
  case kSubtract:
    return evaluated[0] - evaluated[1];
  case kTan:
    return tan(evaluated[0]);
  case kTanh:
    return tanh(evaluated[0]);
  }
}
