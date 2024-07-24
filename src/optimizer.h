#include <cmath>

#include "instruction.h"

class Optimizer {
public:
  void Run(Instruction *instruction) {
    bool changed;
    do {
      changed = false;

      // Rewrites involving binary ops with both operands the same; this should
      // happen BEFORE recursing since there is no point in traversing the whole
      // tree if the root operation is x-x
      if (Arity(instruction->opcode()) == 2 &&
          instruction->operand(0) == instruction->operand(1)) {
        switch (instruction->opcode()) {
        case kAdd:
          VLOG(10) << "x+x --> 2*x";
          changed |= ReplaceInstruction(
              instruction, CreateBinary(kMultiply, instruction->operand(0),
                                        CreateConstant(2)));
        case kDivide:
          VLOG(10) << "x/x --> 1";
          changed |= ReplaceInstruction(instruction, CreateConstant(1));
        case kSubtract:
          VLOG(10) << "x-x --> 0";
          changed |= ReplaceInstruction(instruction, CreateConstant(0));
        default:
          break;
        }
      }

      // TODO Other pruning optimizations combine (without x+x->2x)

      // TODO Put above into HandlePruning

      // Recurse on the instruction's operands before optimizing this one
      for (auto operand : instruction->operands()) {
        Run(operand);
      }

      switch (Arity(instruction->opcode())) {
      case 1:
        changed |= HandleUnary(instruction);
        break;
      case 2:
        changed |= HandleBinary(instruction);
        break;
      }

      // Try for any opcode-specific rewrites
      changed |= HandleOpcode(instruction);
    } while (changed);
  }

  bool HandleUnary(Instruction *unary) {
    Instruction *operand = unary->operand(0);

    // TODO Unify with analogous rewrite in HandleBinary
    if (AllConstantOperands(unary)) {
      VLOG(10) << "Folding unary operation with constant operand";
      return ReplaceInstruction(unary, CreateConstant(Evaluate(unary)));
    }

    if (auto inv_op = Inverse(unary->opcode()); inv_op.has_value()) {
      if (operand->opcode() == Inverse(unary->opcode()).value()) {
        VLOG(10) << "f(g(x)) --> x where f and g are inverses";
        return ReplaceInstruction(unary, operand->operand(0));
      }
    }

    // TODO More general: f(g(x)) --> f(x)
    if (operand->opcode() == kNegate && IsEven(unary->opcode())) {
      VLOG(10) << "f(-x) --> f(x) where f is even";
      return ReplaceInstruction(
          unary, CreateUnary(unary->opcode(), operand->operand(0)));
    }

    // TODO More general: f(g(x)) --> g(f(x))
    if (operand->opcode() == kNegate && IsOdd(unary->opcode())) {
      VLOG(10) << "f(-x) --> -f(x) where f is odd";
      return ReplaceInstruction(
          unary, CreateUnary(kNegate, CreateUnary(unary->opcode(),
                                                  operand->operand(0))));
    }

    // TODO More general: f(g(x)) --> g(x)
    if (unary->opcode() == kAbs && IsNonNegative(operand->opcode())) {
      VLOG(10) << "|f(x)| --> f(x) where f is non-negative";
      return ReplaceInstruction(unary, operand);
    }
    return false;
  }

  bool HandleBinary(Instruction *binary) {
    Instruction *lhs = binary->operand(0);
    Instruction *rhs = binary->operand(1);

    // Fold operations whose every operand is constant
    if (AllConstantOperands(binary)) {
      VLOG(10) << "Folding binary operation with all constant operands";
      return ReplaceInstruction(binary, CreateConstant(Evaluate(binary)));
    }

    // Canonicalize constants to be on the lhs for commutative ops
    if (IsCommutative(binary->opcode()) && rhs->opcode() == kConstant) {
      VLOG(10) << "Canonicalizing constant to lhs of commutative binary op";
      return ReplaceInstruction(binary,
                                CreateBinary(binary->opcode(), rhs, lhs));
    }

    // TODO Fold expressions equal to 0: 0*x, pow(0,x)

    // TODO Fold expressions equal to 1: pow(1,x), pow(x,0), pythag

    // TODO Fold identity elements: x+0, 1*x, pow(x,1)

    // Homomorphism-style rewrites of the form f(g(x),g(y)) --> g(h(x,y))
    if (lhs->opcode() == rhs->opcode()) {
      std::optional<Opcode> new_op;
      if (binary->opcode() == kAdd && lhs->opcode() == kLog) {
        VLOG(10) << "log(x)+log(y) --> log(x*y)";
        new_op = kMultiply;
      } else if (binary->opcode() == kMultiply && lhs->opcode() == kAbs) {
        VLOG(10) << "|x|*|y| --> |x*y|";
        new_op = kMultiply;
      } else if (binary->opcode() == kMultiply && lhs->opcode() == kExp) {
        VLOG(10) << "exp(x)*exp(y) --> exp(x+y)";
        new_op = kAdd;
      } else if (binary->opcode() == kSubtract && lhs->opcode() == kLog) {
        VLOG(10) << "log(x)-log(y) --> log(x/y)";
        new_op = kDivide;
      }

      if (new_op.has_value()) {
        return ReplaceInstruction(
            binary, CreateUnary(lhs->opcode(),
                                CreateBinary(new_op.value(), lhs->operand(0),
                                             rhs->operand(0))));
      }
    }

    // TODO Strength reduction
    return false;
  }

  bool HandleOpcode(Instruction *instruction) {
    switch (instruction->opcode()) {
    case kAdd:
      return HandleAdd(instruction);
    case kDivide:
      return HandleDivide(instruction);
    case kMaximum:
      return HandleMaximum(instruction);
    case kMinimum:
      return HandleMinimum(instruction);
    case kMultiply:
      return HandleMultiply(instruction);
    case kNegate:
      return HandleNegate(instruction);
    case kPower:
      return HandlePower(instruction);
    case kSubtract:
      return HandleSubtract(instruction);
    default:
      return false;
    }
  }

  bool HandleAdd(Instruction *add) {
    assert(add->opcode() == kAdd);

    Instruction *lhs = add->operand(0);
    Instruction *rhs = add->operand(1);

    if (IsConstantWithValue(lhs, 0)) {
      VLOG(10) << "x+0 --> x";
      return ReplaceInstruction(add, rhs);
    }

    // TODO Maybe canonicalize so we don't need both of these?
    if (rhs->opcode() == kNegate) {
      VLOG(10) << "x+(-y) --> x-y";
      return ReplaceInstruction(add,
                                CreateBinary(kSubtract, lhs, rhs->operand(0)));
    }

    if (lhs->opcode() == kNegate) {
      VLOG(10) << "(-x)+y --> y-x";
      return ReplaceInstruction(add,
                                CreateBinary(kSubtract, rhs, lhs->operand(0)));
    }

    // TODO Unify these and make them more general, i.e. xy+zx --> x(y+z)
    if (lhs->opcode() == kMultiply && rhs->opcode() == kMultiply) {
      if (lhs->operand(0) == rhs->operand(0)) {
        VLOG(10) << "xy+xz --> x(y+z)";
        return ReplaceInstruction(
            add,
            CreateBinary(kMultiply, lhs->operand(0),
                         CreateBinary(kAdd, lhs->operand(1), rhs->operand(1))));
      }
      if (lhs->operand(1) == rhs->operand(1)) {
        VLOG(10) << "xz+yz --> (x+y)z";
        return ReplaceInstruction(
            add,
            CreateBinary(kMultiply,
                         CreateBinary(kAdd, lhs->operand(0), rhs->operand(0)),
                         lhs->operand(1)));
      }
    }

    // TODO pow(sin(x),2)+pow(cos(x),2) --> 1
    return false;
  }

  bool HandleDivide(Instruction *divide) {
    assert(divide->opcode() == kDivide);

    Instruction *lhs = divide->operand(0);
    Instruction *rhs = divide->operand(1);

    if (rhs->opcode() == kConstant) {
      VLOG(10) << "x/c --> x*c";
      return ReplaceInstruction(
          divide,
          CreateBinary(kMultiply, lhs, CreateConstant(1 / Evaluate(rhs))));
    }

    if (lhs->opcode() == kSin && rhs->opcode() == kCos &&
        lhs->operand(0) == rhs->operand(0)) {
      VLOG(10) << "sin(x)/cos(x) --> tan(x)";
      return ReplaceInstruction(divide, CreateUnary(kTan, lhs->operand(0)));
    }

    if (rhs->opcode() == kPower) {
      VLOG(10) << "x/pow(y,z) --> x*pow(y,-z)";
      return ReplaceInstruction(
          divide,
          CreateBinary(kMultiply, lhs,
                       CreateBinary(kPower, rhs->operand(0),
                                    CreateUnary(kNegate, rhs->operand(1)))));
    }

    if (lhs->opcode() == kDivide) {
      VLOG(10) << "(x/y)/z --> x/(y*z)";
      return ReplaceInstruction(
          divide, CreateBinary(kDivide, lhs->operand(0),
                               CreateBinary(kMultiply, lhs->operand(1), rhs)));
    }

    if (rhs->opcode() == kDivide) {
      VLOG(10) << "x/(y/z) --> (x*z)/y";
      return ReplaceInstruction(
          divide,
          CreateBinary(kDivide, CreateBinary(kMultiply, lhs, rhs->operand(1)),
                       rhs->operand(0)));
    }
    return false;
  }

  bool HandleMaximum(Instruction *maximum) {
    assert(maximum->opcode() == kMaximum);

    Instruction *lhs = maximum->operand(0);
    Instruction *rhs = maximum->operand(1);

    // TODO Unify monotone rewrites (with sublinear condition? instance of
    // homomorphism?)
    if (lhs->opcode() == rhs->opcode() && IsIncreasing(lhs->opcode())) {
      VLOG(10) << "max(f(x), f(y)) --> f(max(x, y)) where f is increasing";
      return ReplaceInstruction(
          maximum,
          CreateUnary(lhs->opcode(), CreateBinary(kMaximum, lhs->operand(0),
                                                  rhs->operand(0))));
    }

    if (lhs->opcode() == rhs->opcode() && IsDecreasing(lhs->opcode())) {
      VLOG(10) << "max(f(x), f(y)) --> f(min(x, y)) where f is decreasing";
      return ReplaceInstruction(
          maximum,
          CreateUnary(lhs->opcode(), CreateBinary(kMinimum, lhs->operand(0),
                                                  rhs->operand(0))));
    }
    return false;
  }

  bool HandleMinimum(Instruction *minimum) {
    assert(minimum->opcode() == kMinimum);

    Instruction *lhs = minimum->operand(0);
    Instruction *rhs = minimum->operand(1);

    if (lhs->opcode() == rhs->opcode() && IsIncreasing(lhs->opcode())) {
      VLOG(10) << "min(f(x), f(y)) --> f(min(x, y)) where f is increasing";
      return ReplaceInstruction(
          minimum,
          CreateUnary(lhs->opcode(), CreateBinary(kMinimum, lhs->operand(0),
                                                  rhs->operand(0))));
    }

    if (lhs->opcode() == rhs->opcode() && IsDecreasing(lhs->opcode())) {
      VLOG(10) << "min(f(x), f(y)) --> f(max(x, y)) where f is decreasing";
      return ReplaceInstruction(
          minimum,
          CreateUnary(lhs->opcode(), CreateBinary(kMaximum, lhs->operand(0),
                                                  rhs->operand(0))));
    }
    return false;
  }

  bool HandleMultiply(Instruction *multiply) {
    assert(multiply->opcode() == kMultiply);

    Instruction *lhs = multiply->operand(0);
    Instruction *rhs = multiply->operand(1);

    if (IsConstantWithValue(lhs, 0)) {
      VLOG(10) << "0*x --> 0";
      return ReplaceInstruction(multiply, lhs);
    }

    // TODO Unify with 0+x --> x add identity rewrite
    if (IsConstantWithValue(lhs, 1)) {
      VLOG(10) << "1*x --> x";
      return ReplaceInstruction(multiply, rhs);
    }

    if (lhs->opcode() == kPower && rhs->opcode() == kPower) {
      if (lhs->operand(1) == rhs->operand(1)) {
        VLOG(10) << "pow(x,z)*pow(y,z) --> pow(x*y,z)";
        return ReplaceInstruction(
            multiply, CreateBinary(kPower,
                                   CreateBinary(kMultiply, lhs->operand(0),
                                                rhs->operand(0)),
                                   lhs->operand(1)));
      }
      if (lhs->operand(0) == rhs->operand(0)) {
        VLOG(10) << "pow(x,y)*pow(x,z) --> pow(x,y+z)";
        return ReplaceInstruction(
            multiply,
            CreateBinary(kPower, lhs->operand(0),
                         CreateBinary(kAdd, lhs->operand(1), rhs->operand(1))));
      }
    }
    return false;
  }

  bool HandleNegate(Instruction *negate) {
    assert(negate->opcode() == kNegate);

    Instruction *operand = negate->operand(0);

    if (operand->opcode() == kSubtract) {
      VLOG(10) << "-(x-y) --> y-x";
      return ReplaceInstruction(
          negate,
          CreateBinary(kSubtract, operand->operand(1), operand->operand(0)));
    }
    return false;
  }

  bool HandlePower(Instruction *power) {
    assert(power->opcode() == kPower);

    Instruction *lhs = power->operand(0);
    Instruction *rhs = power->operand(1);

    if (IsConstantWithValue(rhs, 0)) {
      VLOG(10) << "pow(x,0) --> 1";
      return ReplaceInstruction(power, CreateConstant(1));
    }

    if (IsConstantWithValue(rhs, 1)) {
      VLOG(10) << "pow(x,1) --> x";
      return ReplaceInstruction(power, lhs);
    }

    if (IsConstantWithValue(lhs, 0)) {
      VLOG(10) << "pow(0,x) --> 0";
      return ReplaceInstruction(power, lhs);
    }

    if (IsConstantWithValue(lhs, 1)) {
      VLOG(10) << "pow(1,x) --> 1";
      return ReplaceInstruction(power, lhs);
    }

    if (lhs->opcode() == kPower) {
      VLOG(10) << "pow(pow(x,y),z) --> pow(x,y*z)";
      return ReplaceInstruction(
          power, CreateBinary(kPower, lhs->operand(0),
                              CreateBinary(kMultiply, lhs->operand(1), rhs)));
    }
    return false;
  }

  bool HandleSubtract(Instruction *subtract) {
    assert(subtract->opcode() == kSubtract);

    // TODO pow(x,2)-pow(y,2) --> (x-y)*(x+y)
    return false;
  }
};
