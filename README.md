This is my implementation of an XLA-like optimizer for mathematical computations. I hope to eventually use it to generate CUDA code for executing functions defined by user-input LaTeX.

Right now I need to work on:

- Make Constant (and Variable? and Rng?) a subclass of Instruction with a field for data. Needs a function to retrieve the value. After that, implement constant related optimization passes
- Improve ReplaceInstruction and Optimize function. We should recursively optimize until no changes occur
- Better printing - fix opcode_to_string syntax
- Long term: better pattern matcher
