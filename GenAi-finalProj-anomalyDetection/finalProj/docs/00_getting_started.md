# 00 — Getting Started (read this first)

This guide assumes you have written little or no Python before. Take it slowly;
each step is small. By the end you'll be able to run the project and see results.
You do **not** need to understand the deep-learning math to get the code running —
that comes later, in your own guide.

## What you are building (one paragraph)
We train a model on *normal* pictures only. A good model learns to recreate
normal pictures well. When we later show it a *weird* picture, it recreates it
badly. "How badly it recreates the picture" becomes an **anomaly score** — high
score means "this looks abnormal." We do this with three kinds of model and
compare them. Nicolás built the shared machinery and one model (the
autoencoder). You build one of the other two.

## Step 1 — Install Python
- Download Python 3.11 from <https://www.python.org/downloads/>. During install
  on Windows, **tick "Add Python to PATH."**
- Check it worked. Open a terminal (Windows: "PowerShell"; Mac: "Terminal") and type:
  ```bash
  python --version
  ```
  You should see something like `Python 3.11.x`. (On some Macs use `python3`.)

## Step 2 — Get the project and open a terminal *inside* it
- Unzip the project folder somewhere easy, e.g. your Desktop.
- In the terminal, move into the folder. Example:
  ```bash
  cd Desktop/genai-anomaly-detection
  ```
  `cd` means "change directory." If the path has spaces, wrap it in quotes.

## Step 3 — Make a clean workspace (virtual environment)
A "virtual environment" is a private box for this project's libraries so they
don't clash with anything else on your computer. Create and activate it:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate
```
When it's active you'll see `(.venv)` at the start of your terminal line.
You repeat the *activate* line every time you open a new terminal.

## Step 4 — Install the libraries
```bash
pip install -r requirements.txt
```
This downloads PyTorch (the deep-learning library) and a few others. It can take
several minutes and download a lot — that's normal. Do it once.

## Step 5 — Get the data
The code tries to download Fashion-MNIST automatically the first time you run it.
If that fails (some networks block the download), run these four lines from
inside the project folder to fetch it manually:
```bash
mkdir -p data/FashionMNIST/raw && cd data/FashionMNIST/raw
# (Windows PowerShell users: see REPRODUCIBILITY.md for the equivalent commands)
BASE="https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion"
for f in train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz; do curl -sSL -o $f "$BASE/$f"; done
cd ../../..
```

## Step 6 — Run the working model to prove your setup works
```bash
python scripts/run_experiment.py --model autoencoder --epochs 5
```
You should see the loss number going **down** each epoch, then a line like
`autoencoder: AUROC=0.91 ...`. Open the `results/` folder — there are new images
(ROC curve, score histogram). **If you got here, your environment is correct.**
Congratulations, that was the hard part.

## How to ask for help efficiently
When something breaks, copy the **whole** error message (the red text), the
command you ran, and your operating system. The last line of a Python error is
usually the real clue (e.g. `ModuleNotFoundError: No module named 'torch'` means
the install in Step 4 didn't finish or the venv isn't active).

## Tiny terminal cheat-sheet
| Command | Meaning |
|---|---|
| `cd folder` | go into a folder |
| `cd ..` | go up one folder |
| `ls` (Mac/Linux) / `dir` (Windows) | list files here |
| `python script.py` | run a Python file |
| `Ctrl + C` | stop a running program |

Next: read `01_how_the_code_works.md` to understand the pieces, then open the
guide for your model (`guide_person2_vae.md` or `guide_person3_aae.md`).
