#include <string>

enum Opcode {
  kRandom,
  kConstant,
  kVariable,

  kSin,
  kTanh,
  kRelu,
  kExp,
  kNeg,

  kAdd,
  kMul,
  kDiv
};

int Arity(Opcode opcode) {
  switch (opcode) {
  case kRandom:
  case kConstant:
  case kVariable:
    return 0;
  case kSin:
  case kTanh:
  case kRelu:
  case kExp:
  case kNeg:
    return 1;
  case kAdd:
  case kMul:
  case kDiv:
    return 2;
  }
}

std::string opcode_to_string(Opcode opcode) {
  switch (opcode) {
  case kRandom:
    return "random";
  case kConstant:
    return "constant";
  case kVariable:
    return "variable";
  case kSin:
    return "sin";
  case kTanh:
    return "tanh";
  case kRelu:
    return "relu";
  case kExp:
    return "exp";
  case kNeg:
    return "neg";
  case kAdd:
    return "add";
  case kMul:
    return "mul";
  case kDiv:
    return "div";
  }
}
