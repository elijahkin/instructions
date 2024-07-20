## Overview
This is my implementation of an XLA-like optimizer for mathematical computations. I hope to eventually use it to generating CUDA code from user-input LaTeX.

Given a computation, this optimizer constructs and traverses a tree representation of it, applying algebraic properties on its subtrees to produce an equivalent but (hopefully) cheaper computation.

For example,
```c++
parameter.0 = parameter()
exp.2 = exp(parameter.0)
parameter.1 = parameter()
exp.3 = exp(parameter.1)
multiply.4 = multiply(exp.2, exp.3)
```
is automatically optimized into
```c++
parameter.0 = parameter()
parameter.1 = parameter()
add.5 = add(parameter.0, parameter.1)
exp.6 = exp(add.5)
```

## To-Do List
- Make CreateParameter accept an index?
- Store the double passed in to CreateConstant, and add getter function
- Might need to make Rng be Binary?
- Stylistic/syntax improvements
- Long term: a nicer pattern matcher
- Evaluation
