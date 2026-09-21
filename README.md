# Prodigy-GA-3

# 🔗 Task-03: Text Generation with Markov Chains

A text generator built from scratch in pure Python using **Markov chains**. It learns which word (or character) tends to follow the previous one(s) and samples new text from those probabilities.

Completed as part of the **Generative AI Internship at Prodigy InfoTech**.

---

## 🎯 Objective
Implement a simple text generation algorithm using Markov chains: a statistical model that predicts the probability of a character or word based on the previous one(s).

## 🧠 How It Works
A Markov chain assumes the next token depends only on the last **n** tokens (the *state*), not on the whole history.

1. **Training:** slide over the text and count how often each token follows each state.
2. **Probabilities:** divide the counts by the total for that state.
3. **Generation:** start from a state, pick the next token in proportion to those probabilities, shift the state forward, and repeat.

**Example** (order 1, text: `the cat sat on the mat the cat ate the fish`):

| State | Next token probabilities |
|---|---|
| the | cat: 0.50, mat: 0.25, fish: 0.25 |
| cat | sat: 0.50, ate: 0.50 |
| sat | on: 1.00 |

## 🛠️ Implementation
- One `MarkovChain` class, written from scratch with only the Python standard library (`random`, `re`, `collections`)
- **Word-level and character-level** modes
- **Configurable order** (how many previous tokens form the state)
- Optional **start prompt** and fixed **seed** for reproducible output
- Dead-end handling: if a state has no known next token, the model jumps to a random state
- Dataset: Tiny Shakespeare (about 1.1 million characters, roughly 200,000 words)

## 📊 Results

### Model size grows with order
| Level | Order | States learned |
|---|---|---|
| Word | 1 | 14,565 |
| Word | 2 | 100,615 |
| Word | 3 | 192,553 |
| Character | 2 | 1,403 |
| Character | 4 | 50,712 |
| Character | 7 | 447,352 |

### Sample output
**Word-level, order 3**, prompt `the king is`:
```
the king is left behind,
And these, who often drown'd could never die,
With lies well steel'd with weighty arguments;
And for his meed, poor lord, he doth it publicly,
```

**Character-level, order 2** (too little context, so mostly invented words):
```
Maram th ow ifen wreatentand thard,
In any bithe for tree, so;
```

**Character-level, order 7** (real words and sentence shapes, but heavily copied from the source):
```
or newer torture the lands and let them make I as patience:
But let him such was Prince Edward's mockery king by our tribunes forth;
```

Outputs shown come from seed 42 and may differ slightly on your machine.

## 💡 Key Learnings
- **Low order means randomness.** With little context, the model produces text that has the right vocabulary but little structure.
- **High order means fluency, but also copying.** Most states at high orders have only one possible next token, so the model largely reproduces training text instead of creating new text.
- **Character-level chains need a higher order** than word-level ones to form real words.
- **The state space explodes.** The number of states grows quickly with order, so memory use grows too.

## ⚠️ Limitations
- A Markov chain has no memory beyond its order, so it cannot keep a topic, plot or argument going across a paragraph.
- It can only produce combinations it has already seen in the training data.
- It is the statistical ancestor of modern language models. Compared with transformers like GPT-2 (my Task-01), it has no learned representation of meaning.

## ▶️ How to Run
1. Open `markov_chain_text_generation.ipynb` in [Google Colab](https://colab.research.google.com) (no GPU needed).
2. Run all cells in order. It finishes in about a minute.
3. To use your own text, upload a `.txt` file and change `path` in the dataset cell.

Quick usage:
```python
model = MarkovChain(order=2, level="word").train(text)
print(model.generate(length=60, start="the king", seed=42))
```

## 📁 Project Structure
```
├── markov_chain_text_generation.ipynb
└── README.md
```

## 📚 References
- Markov chains: [Wikipedia](https://en.wikipedia.org/wiki/Markov_chain)
- Tiny Shakespeare dataset from Andrej Karpathy's [char-rnn](https://github.com/karpathy/char-rnn)
