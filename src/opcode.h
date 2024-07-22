#include <string>

enum Opcode {
  kAbs,
  kAdd,
  kAtan2,
  kConstant,
  kCos,
  kDivide,
  kExp,
  kLog,
  kMaximum,
  kMinimum,
  kMultiply,
  kNegate,
  kParameter,
  kPower,
  kRelu,
  kRng,
  kSin,
  kSubtract,
  kTan,
  kTanh
};

int Arity(Opcode opcode) {
  switch (opcode) {
  case kConstant:
  case kParameter:
  case kRng:
    return 0;
  case kAbs:
  case kCos:
  case kExp:
  case kLog:
  case kNegate:
  case kRelu:
  case kSin:
  case kTan:
  case kTanh:
    return 1;
  case kAdd:
  case kAtan2:
  case kDivide:
  case kMaximum:
  case kMinimum:
  case kMultiply:
  case kPower:
  case kSubtract:
    return 2;
  }
}

std::string opcode_to_string(Opcode opcode) {
  switch (opcode) {
  case kAbs:
    return "abs";
  case kAdd:
    return "add";
  case kAtan2:
    return "atan2";
  case kConstant:
    return "constant";
  case kCos:
    return "cosine";
  case kDivide:
    return "divide";
  case kExp:
    return "exp";
  case kLog:
    return "log";
  case kMaximum:
    return "maximum";
  case kMinimum:
    return "minimum";
  case kMultiply:
    return "multiply";
  case kNegate:
    return "negate";
  case kParameter:
    return "parameter";
  case kPower:
    return "power";
  case kRelu:
    return "relu";
  case kRng:
    return "rng";
  case kSin:
    return "sine";
  case kSubtract:
    return "subtract";
  case kTan:
    return "tan";
  case kTanh:
    return "tanh";
  }
}
