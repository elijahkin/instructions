#include <string>

enum Opcode {
  kAbs,
  kAdd,
  kAtan2,
  kConstant,
  kCos,
  kDivide,
  kExp,
  kImag,
  kLog,
  kMaximum,
  kMinimum,
  kMultiply,
  kNegate,
  kPower,
  kReal,
  kRelu,
  kRemainder,
  kRng,
  kSin,
  kSubtract,
  kTan,
  kTanh,
  kVariable
};

int Arity(Opcode opcode) {
  switch (opcode) {
  case kConstant:
  case kRng:
  case kVariable:
    return 0;
  case kAbs:
  case kAtan2:
  case kCos:
  case kExp:
  case kImag:
  case kLog:
  case kNegate:
  case kReal:
  case kRelu:
  case kSin:
  case kTan:
  case kTanh:
    return 1;
  case kAdd:
  case kDivide:
  case kMaximum:
  case kMinimum:
  case kMultiply:
  case kPower:
  case kRemainder:
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
  case kImag:
    return "imag";
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
  case kPower:
    return "power";
  case kReal:
    return "real";
  case kRelu:
    return "relu";
  case kRemainder:
    return "remainder";
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
  case kVariable:
    return "variable";
  }
}
