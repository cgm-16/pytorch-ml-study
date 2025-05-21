# Softmax Regression

Softmax regression is a generalization of logistic regression that allows us to handle multi-class classification problems, rather than just binary classification.

## Relationship to Logistic Regression

1. **Extension of Binary to Multi-class:**
   - Logistic regression uses the sigmoid function to output a probability between 0 and 1 for binary classification
   - Softmax regression extends this concept to multiple classes by outputting a probability distribution across all classes

2. **Mathematical Formulation:**
   - In logistic regression, we use: $P(y=1|x) = \frac{1}{1 + e^{-z}}$ where $z = wx + b$
   - In softmax regression, we use: $P(y=j|x) = \frac{e^{z_j}}{\sum_{k=1}^{K} e^{z_k}}$ where $z_j = w_j x + b_j$ for each class $j$

3. **Key Properties:**
   - Outputs sum to 1 (valid probability distribution)
   - Each output is between 0 and 1
   - Exponential function ensures all values are positive

