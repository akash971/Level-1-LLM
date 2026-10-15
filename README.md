# NextWord AI — Level-1 Language Model & Bigram Playground

An interactive, educational implementation of a **Bigram Language Model (Markov Chain)** in both pure Python and modern Vanilla Web technologies (HTML5, CSS3, JavaScript).

NextWord AI demonstrates the foundational mechanics behind modern Large Language Models (LLMs): **tokenization, frequency transition counting, probability distributions, and autoregressive next-word prediction.**

---

## 🌟 Features

- **Pure Python CLI Model**:
  - Heavily commented, zero-dependency Python script (`next_word_predictor.py`).
  - Reads and cleans text from `sample.txt`.
  - Builds a transition frequency dictionary `{ current_word: { next_word: count } }`.
  - Generates 5-word sentences based on word frequencies.

- **Interactive Web Playground**:
  - **Live Corpus Editor**: Edit training text on the fly or choose between built-in presets (*Animals*, *Sci-Fi Space*, *Dev Life*).
  - **The "Brain" Inspector**: Visualizes frequency tallies, votes, and percentage probabilities for every candidate next word.
  - **Sentence Generator**: Generates sentences with adjustable lengths (3–15 words).
  - **Step-by-Step Play Mode**: Animates token-by-token generation to visualize how autoregressive prediction works.
  - **Greedy vs. Creative Sampling**: Toggle between Greedy (highest frequency) and Probabilistic (weighted random sampling).
  - **ELI5 Guide**: Built-in "Explain Like I'm 5" modal explaining tokenization, bigrams, and chain prediction.

---

## 📁 Repository Structure

```text
Level-1-LLM/
├── sample.txt              # Sample training text corpus
├── next_word_predictor.py  # Pure Python Bigram model & CLI generator
├── index.html              # Modern web UI structure
├── style.css               # Dark mode, glassmorphism design system
├── app.js                  # Client-side NLP tokenization & interactive engine
├── server.py               # Lightweight local HTTP server launcher
├── .gitignore              # Ignored files
└── README.md               # Project documentation
```

---

## 🚀 Quick Start

### Option A: Run the Python CLI

```bash
python next_word_predictor.py
```

Example interaction:
```text
Enter a starting word: quick
Generated 5-word sentence:
  -> Quick brown fox jumps over.
```

### Option B: Run the Web App in Your Browser

1. Start the local server:
   ```bash
   python server.py
   ```
2. Or simply open `index.html` directly in any web browser!

---

## 🧠 How It Works (ELI5)

1. **✂️ Tokenize (Scissors)**: The text is stripped of punctuation and split into lowercase word stickers called *tokens*.
2. **📝 Tally Notebook (Bigrams)**: For every pair of adjacent words `(word[i], word[i+1])`, the model adds a tally mark in its memory:
   ```python
   {
       "quick": {"brown": 2, "rabbit": 1},
       "brown": {"fox": 3}
   }
   ```
3. **🏆 Pick the Winner**: When given a starting word, the model checks that page in the notebook and selects the word with the highest votes.
4. **🔗 Chain the Sentence**: The chosen word becomes the new starting word, repeating the process 5 times to form a complete sentence.

---

## 📄 License

MIT License. Free to use and learn from!
