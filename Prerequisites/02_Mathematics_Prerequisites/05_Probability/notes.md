# Mathematics Prerequisites: 05. Probability & Bayes' Theorem for Machine Learning

Deterministic programming says: *"If X happens, Y definitely happens."*  
Machine Learning deals with the real world, where data is noisy and incomplete. ML models say: *"Given evidence X, there is an 87% probability that Y is true."*

Probability is the mathematical framework for **measuring uncertainty**.

---

## 1. The Core Vocabulary of Probability

- **Experiment:** An action or process that leads to an uncertain outcome (e.g., flipping a coin, rolling a die, a user visiting a website).
- **Outcome:** A single possible result of an experiment (e.g., getting a "Heads").
- **Sample Space ($S$):** The set of **all possible outcomes**.
  - For a coin toss: $S = \{\text{Heads}, \text{Tails}\}$ (Total = 2)
  - For a 6-sided die: $S = \{1, 2, 3, 4, 5, 6\}$ (Total = 6)
- **Event ($A$):** A specific outcome or combination of outcomes we are interested in (e.g., "rolling an even number").

---

## 2. Calculating Basic Probability

The probability of an event $A$, written $P(A)$, is:
$$P(A) = \frac{\text{Number of favorable outcomes}}{\text{Total number of possible outcomes in } S}$$

### The Universal Probability Scale:
$$0.0 \le P(A) \le 1.0$$
- $P(A) = 0.0$: The event is **impossible** (e.g., rolling an 8 on a standard 6-sided die).
- $P(A) = 1.0$: The event is **guaranteed / certain** to happen.
- All probabilities sum to 1: $P(\text{Event}) + P(\text{Not Event}) = 1.0$.

### Small Examples:
1. **Coin Toss:**
   $$P(\text{Heads}) = \frac{1}{2} = 0.50 \quad (50\%)$$
2. **Rolling an Even Number on a Die:**
   - Favorable outcomes: $\{2, 4, 6\}$ (Count = 3)
   - Sample space: $\{1, 2, 3, 4, 5, 6\}$ (Count = 6)
   $$P(\text{Even}) = \frac{3}{6} = 0.50 \quad (50\%)$$

---

## 3. Independent vs. Dependent Events

- **Independent Events:** The occurrence of Event A has **zero impact** on the probability of Event B.
  - *Example:* Flipping Heads on Coin 1 does not change the chances of flipping Heads on Coin 2.
  - **Multiplication Rule:** $P(A \text{ and } B) = P(A) \times P(B)$
  - *Example:* $P(\text{Heads and Heads}) = 0.5 \times 0.5 = 0.25$ ($25\%$).
- **Dependent Events:** The occurrence of Event A **changes the probability** of Event B.
  - *Example:* Drawing a card from a deck without replacing it changes the probabilities for the next draw.

---

## 4. Conditional Probability: $P(A \mid B)$

Conditional probability is the probability of Event $A$ occurring **given that Event $B$ has already occurred**.

$$\text{Notation: } P(A \mid B) \quad \text{(Read: "Probability of A given B")}$$

### Intuition: Shrinking the Sample Space
When you learn that Event $B$ has already happened, your universe of possibilities **shrinks** down to only $B$.

### Formula:
$$P(A \mid B) = \frac{P(A \cap B)}{P(B)}$$
- $P(A \cap B)$: The probability that **both** $A$ and $B$ happen together (Joint Probability).
- $P(B)$: The probability of the condition $B$.

### Real-World Example:
Suppose in a school of 100 students:
- $30$ students are in the Coding Club ($B$).
- $15$ students are in BOTH Coding Club and Math Club ($A \cap B$).

What is the probability a student is in Math Club **given** that they are in Coding Club?
$$P(\text{Math} \mid \text{Coding}) = \frac{15}{30} = \mathbf{0.50 \quad (50\%)}$$

---

## 5. Bayes' Theorem: Updating Beliefs with Evidence

Bayes' Theorem is one of the most celebrated equations in modern machine learning. It tells us **how to update our initial beliefs about the world when new evidence arrives**.

$$\text{Prior Belief} + \text{New Evidence} \longrightarrow \text{Updated Posterior Belief}$$

### The Formula:
$$P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B)}$$

### Symbol-by-Symbol Breakdown:
| Symbol | Formal Name | Plain English Meaning |
| :--- | :--- | :--- |
| **$P(A \mid B)$** | **Posterior** | What is the probability hypothesis $A$ is true **after** seeing evidence $B$? |
| **$P(B \mid A)$** | **Likelihood** | How likely would we see evidence $B$ if hypothesis $A$ were true? |
| **$P(A)$** | **Prior** | What was our baseline belief that $A$ is true **before** seeing any evidence? |
| **$P(B)$** | **Marginal Evidence** | Total probability of observing evidence $B$ across all possibilities. |

---

## 6. A Step-by-Step Numerical Example of Bayes' Theorem

Let's test an email spam filter:
- We want to know: What is the probability an email is **Spam** ($A$) given that it contains the word **"FREE"** ($B$)?

### The Given Probabilities:
1. **Prior $P(\text{Spam})$:** $20\%$ of all incoming emails are spam $\implies P(A) = 0.20$.
2. **Likelihood $P(\text{"FREE"} \mid \text{Spam})$:** $80\%$ of spam emails contain the word "FREE" $\implies P(B \mid A) = 0.80$.
3. **Evidence $P(\text{"FREE"})$:** Across all emails (spam and non-spam), the word "FREE" appears in $25\%$ of messages $\implies P(B) = 0.25$.

### The Calculation:
$$P(\text{Spam} \mid \text{"FREE"}) = \frac{P(\text{"FREE"} \mid \text{Spam}) \cdot P(\text{Spam})}{P(\text{"FREE"})}$$

Substitute the numbers:
$$P(\text{Spam} \mid \text{"FREE"}) = \frac{0.80 \times 0.20}{0.25} = \frac{0.16}{0.25} = \mathbf{0.64 \quad (64\%)}$$

### The Plain English Interpretation:
Before looking at the words in the email, the baseline chance it was spam was only $20\%$.  
Once we discovered the word "FREE" inside it, our updated belief (posterior) jumped to **$64\%$**!

---

## 7. Connecting Probability to Machine Learning Algorithms

| Machine Learning Topic | How It Uses Probability |
| :--- | :--- |
| **Naive Bayes Classifier** | Directly applies Bayes' Theorem assuming features (words) are conditionally independent to classify text. |
| **Logistic Regression** | Models the log-odds of a class and uses the Sigmoid function to output $P(y = 1 \mid X)$. |
| **Decision Trees** | Uses class probabilities $p_i$ at each split to calculate **Entropy** and **Gini Impurity**. |
| **Classification Thresholds** | Deciding whether to classify a transaction as Fraud if $P(\text{Fraud}) > 0.50$ (or $0.10$ for high-risk systems). |
| **Softmax Activation** | Converts raw neural network outputs into a normalized probability distribution where $\sum p_i = 1.0$. |
