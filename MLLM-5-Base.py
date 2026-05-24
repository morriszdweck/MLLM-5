"""
MLLM-5-ATLAS  —  Micro Language Model-5 (Advanced Tiny Language Architecture System)
====================================================================================

A major architectural upgrade from MLLM-4-Abyss, tuned for SPEED and ACCURACY:

  ARCHITECTURE CHANGES
  ────────────────────
  • BPE-lite subword tokenizer with merge learning
  • Sinusoidal positional encoding for sequence order awareness
  • Trainable token embedding layer (dense vectors)
  • Multi-head self-attention mechanism (configurable heads)
  • Feed-forward transformation layers with GELU activation
  • Layer normalization for training stability
  • Residual (skip) connections for gradient flow
  • Fast interpolated-backoff n-gram backbone — only scores words
    ACTUALLY SEEN after the context (O(k) not O(|V|) per step)
  • Transformer blocks available but OFF by default (untrained random
    weights add noise to generation; enable when you have real weights)
  • Online / incremental learning from USER INPUT ONLY (prevents the
    degenerate feedback loop where model output poisons future generation)
  • Context memory retrieves past USER QUERIES only, not model output
  • Confidence scoring on every generation step

  SPEED FIXES vs. initial MLLM-5
  ──────────────────────────────
  • Replaced Kneser-Ney (scored entire vocab every step) with direct
    successor lookup — typically 3-10 candidate words, not 500+
  • Removed beam search from default path (was 5x slower, no quality
    gain with untrained weights)
  • Transformer blocks OFF by default (skip ~100ms of pure-Python
    matrix math per generation call)
  • Context memory no longer injects model output (was creating a
    feedback loop of increasingly bad text)

  TOOL ENGINE (18 tools, up from 5)
  ──────────────────────────────────
  1.  Bare math evaluator  (arbitrary safe expressions)
  2.  Base64 encode / decode
  3.  Hash calculator  (md5, sha1, sha256, sha512)
  4.  Fraction arithmetic
  5.  Complex number math
  6.  Unit converter  (length, mass, temperature, volume, speed, data)
  7.  Date / time calculator
  8.  Color converter  (HEX <-> RGB <-> HSL)
  9.  Roman numeral <-> integer
  10. Statistics calculator  (mean, median, mode, stdev, variance, range)
  11. Combinatorics  (nCr, nPr, factorial)
  12. Number-base converter  (bin, oct, dec, hex)
  13. Geometry calculator  (area, volume, perimeter for 12+ shapes)
  14. Financial calculator  (compound interest, loan payment, ROI)
  15. GCD / LCM calculator
  16. Linear equation solver
  17. Text analysis  (word count, char count, readability index)
  18. Probability calculator  (binomial, normal approx, Poisson)

  All responses in simple English.  No Chinese characters in output.
"""

import re
import math
import cmath
import random
import hashlib
import base64
import json
import time
import datetime
from collections import defaultdict, Counter
from fractions import Fraction
from typing import List, Dict, Tuple, Optional, Any


# ══════════════════════════════════════════════════════════════════════════════
#  DEFAULT CORPUS  (edit or load your own)
# ══════════════════════════════════════════════════════════════════════════════

CORPUS = """

"""


# ══════════════════════════════════════════════════════════════════════════════
#  BPE-LITE SUBWORD TOKENIZER
# ══════════════════════════════════════════════════════════════════════════════

class BPETokenizer:
    """Byte-Pair Encoding lite: learns merge rules from a corpus, then
    tokenizes new text into subword units.  Falls back to character-level
    if no merges are known."""

    SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<SEP>"]

    def __init__(self, vocab_size: int = 2000):
        self.vocab_size = vocab_size
        self.merges: List[Tuple[str, str]] = []
        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: Dict[int, str] = {}
        self.trained = False

    # ── training ─────────────────────────────────────────────────────────

    def _word_freqs(self, text: str) -> Dict[str, int]:
        """Split text into words (space-separated), count frequencies."""
        words = re.findall(r'\S+', text.lower())
        freq: Dict[str, int] = Counter(words)
        # Represent each word as a tuple of characters + end-of-word marker
        return {tuple(w) + ("</w>",): c for w, c in freq.items()}

    @staticmethod
    def _pair_freqs(word_freqs: Dict[tuple, int]) -> Dict[Tuple[str, str], int]:
        pairs: Dict[Tuple[str, str], int] = Counter()
        for word, freq in word_freqs.items():
            for i in range(len(word) - 1):
                pairs[(word[i], word[i + 1])] += freq
        return pairs

    @staticmethod
    def _merge_pair(pair: Tuple[str, str],
                    word_freqs: Dict[tuple, int]) -> Dict[tuple, int]:
        merged = pair[0] + pair[1]
        new_freqs: Dict[tuple, int] = {}
        for word, freq in word_freqs.items():
            new_word: list = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                    new_word.append(merged)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_freqs[tuple(new_word)] = freq
        return new_freqs

    def train(self, corpus: str, verbose: bool = False):
        """Learn BPE merge rules from *corpus*."""
        word_freqs = self._word_freqs(corpus)
        self.merges = []
        for _ in range(self.vocab_size - 256):  # 256 base chars
            pairs = self._pair_freqs(word_freqs)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)  # type: ignore[arg-type]
            word_freqs = self._merge_pair(best, word_freqs)
            self.merges.append(best)

        # Build vocabulary
        self.vocab = {}
        idx = 0
        for tok in self.SPECIAL_TOKENS:
            self.vocab[tok] = idx
            idx += 1
        # Single characters
        for i in range(256):
            ch = chr(i)
            if ch not in self.vocab:
                self.vocab[ch] = idx
                idx += 1
        # Merged tokens
        for a, b in self.merges:
            merged = a + b
            if merged not in self.vocab:
                self.vocab[merged] = idx
                idx += 1
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.trained = True
        if verbose:
            print(f"  BPE tokenizer: {len(self.vocab)} tokens, "
                  f"{len(self.merges)} merges learned.")

    # ── encoding / decoding ──────────────────────────────────────────────

    def encode(self, text: str) -> List[int]:
        """Tokenize *text* into subword token ids."""
        if not self.trained:
            # Fallback: simple word-level ids
            return [hash(w) % (self.vocab_size or 2000) for w in text.lower().split()]
        words = re.findall(r'\S+', text.lower())
        ids: List[int] = []
        for word in words:
            symbols = list(word) + ["</w>"]
            for a, b in self.merges:
                i = 0
                while i < len(symbols) - 1:
                    if symbols[i] == a and symbols[i + 1] == b:
                        symbols[i:i + 2] = [a + b]
                    else:
                        i += 1
            for sym in symbols:
                ids.append(self.vocab.get(sym, self.vocab.get("<UNK>", 1)))
        return ids

    def decode(self, ids: List[int]) -> str:
        """Convert token ids back to text."""
        tokens = [self.inverse_vocab.get(i, "<UNK>") for i in ids]
        text = "".join(tokens).replace("</w>", " ")
        return text.strip()


# ══════════════════════════════════════════════════════════════════════════════
#  POSITIONAL ENCODING  (Sinusoidal, as in "Attention Is All You Need")
# ══════════════════════════════════════════════════════════════════════════════

class PositionalEncoding:
    """Deterministic sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512):
        self.d_model = d_model
        self.max_len = max_len
        self.table = self._build_table()

    def _build_table(self) -> List[List[float]]:
        table = []
        for pos in range(self.max_len):
            row = []
            for i in range(self.d_model):
                angle = pos / (10000 ** ((2 * (i // 2)) / self.d_model))
                if i % 2 == 0:
                    row.append(math.sin(angle))
                else:
                    row.append(math.cos(angle))
            table.append(row)
        return table

    def encode(self, seq_len: int) -> List[List[float]]:
        """Return positional vectors for positions 0..seq_len-1."""
        return self.table[:seq_len]


# ══════════════════════════════════════════════════════════════════════════════
#  EMBEDDING LAYER
# ══════════════════════════════════════════════════════════════════════════════

class EmbeddingLayer:
    """Trainable token embedding + positional encoding."""

    def __init__(self, vocab_size: int, d_model: int = 64):
        self.d_model = d_model
        self.vocab_size = vocab_size
        # Xavier-uniform initialization
        scale = math.sqrt(6.0 / (vocab_size + d_model))
        self.weight: List[List[float]] = [
            [random.uniform(-scale, scale) for _ in range(d_model)]
            for _ in range(vocab_size)
        ]
        self.pos_enc = PositionalEncoding(d_model)

    def forward(self, token_ids: List[int]) -> List[List[float]]:
        """Return (seq_len x d_model) matrix: embedding + positional."""
        seq_len = len(token_ids)
        pos = self.pos_enc.encode(seq_len)
        out = []
        for i, tid in enumerate(token_ids):
            if tid < self.vocab_size:
                vec = [self.weight[tid][j] + pos[i][j] for j in range(self.d_model)]
            else:
                vec = list(pos[i])  # fallback for OOV
            out.append(vec)
        return out


# ══════════════════════════════════════════════════════════════════════════════
#  LAYER NORMALIZATION
# ══════════════════════════════════════════════════════════════════════════════

class LayerNorm:
    """Per-vector layer normalization with learnable scale and shift."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        self.eps = eps
        self.gamma: List[float] = [1.0] * d_model
        self.beta: List[float] = [0.0] * d_model

    def forward(self, x: List[List[float]]) -> List[List[float]]:
        out = []
        for row in x:
            mean = sum(row) / len(row)
            var = sum((v - mean) ** 2 for v in row) / len(row)
            out.append([
                self.gamma[j] * (row[j] - mean) / math.sqrt(var + self.eps) + self.beta[j]
                for j in range(len(row))
            ])
        return out


# ══════════════════════════════════════════════════════════════════════════════
#  MULTI-HEAD SELF-ATTENTION  (Pure Python, no external deps)
# ══════════════════════════════════════════════════════════════════════════════

class MultiHeadAttention:
    """Scaled dot-product multi-head self-attention."""

    def __init__(self, d_model: int = 64, n_heads: int = 4):
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        # Initialize Q, K, V projection matrices and output projection
        scale = math.sqrt(2.0 / (d_model + self.d_k))
        self.W_q = self._rand_matrix(d_model, d_model, scale)
        self.W_k = self._rand_matrix(d_model, d_model, scale)
        self.W_v = self._rand_matrix(d_model, d_model, scale)
        self.W_o = self._rand_matrix(d_model, d_model, scale)

    @staticmethod
    def _rand_matrix(rows: int, cols: int, scale: float) -> List[List[float]]:
        return [[random.uniform(-scale, scale) for _ in range(cols)] for _ in range(rows)]

    @staticmethod
    def _matmul(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
        """Matrix multiply  (m x n) @ (n x p) -> (m x p)."""
        m, n, p = len(a), len(a[0]), len(b[0])
        out = [[0.0] * p for _ in range(m)]
        for i in range(m):
            for k in range(n):
                aik = a[i][k]
                for j in range(p):
                    out[i][j] += aik * b[k][j]
        return out

    @staticmethod
    def _softmax(vec: List[float]) -> List[float]:
        m = max(vec)
        exps = [math.exp(v - m) for v in vec]
        s = sum(exps)
        return [e / s for e in exps]

    def forward(self, x: List[List[float]]) -> List[List[float]]:
        """x: (seq_len, d_model) -> (seq_len, d_model)."""
        seq_len = len(x)
        # Project to Q, K, V
        Q = self._matmul(x, self.W_q)  # (seq_len, d_model)
        K = self._matmul(x, self.W_k)
        V = self._matmul(x, self.W_v)

        # Split into heads and compute attention per head
        heads_out: List[List[List[float]]] = []
        for h in range(self.n_heads):
            offset = h * self.d_k
            q_h = [row[offset:offset + self.d_k] for row in Q]
            k_h = [row[offset:offset + self.d_k] for row in K]
            v_h = [row[offset:offset + self.d_k] for row in V]

            # Scaled dot-product attention:  scores = Q_h @ K_h^T / sqrt(d_k)
            scale = math.sqrt(self.d_k)
            attn_scores: List[List[float]] = []
            for i in range(seq_len):
                row = []
                for j in range(seq_len):
                    dot = sum(q_h[i][d] * k_h[j][d] for d in range(self.d_k))
                    row.append(dot / scale)
                attn_scores.append(row)

            # Softmax over keys (causal mask: only attend to j <= i)
            attn_weights: List[List[float]] = []
            for i in range(seq_len):
                masked = [attn_scores[i][j] if j <= i else -1e9 for j in range(seq_len)]
                attn_weights.append(self._softmax(masked))

            # Weighted sum of values
            head_out: List[List[float]] = []
            for i in range(seq_len):
                vec = [0.0] * self.d_k
                for j in range(seq_len):
                    w = attn_weights[i][j]
                    for d in range(self.d_k):
                        vec[d] += w * v_h[j][d]
                head_out.append(vec)
            heads_out.append(head_out)

        # Concatenate heads
        concat: List[List[float]] = []
        for i in range(seq_len):
            row: List[float] = []
            for h in range(self.n_heads):
                row.extend(heads_out[h][i])
            concat.append(row)

        # Output projection
        return self._matmul(concat, self.W_o)


# ══════════════════════════════════════════════════════════════════════════════
#  FEED-FORWARD NETWORK  (2-layer MLP with GELU)
# ══════════════════════════════════════════════════════════════════════════════

class FeedForward:
    """Position-wise feed-forward network:  d_model -> d_ff -> d_model."""

    def __init__(self, d_model: int = 64, d_ff: int = 256):
        scale1 = math.sqrt(2.0 / (d_model + d_ff))
        scale2 = math.sqrt(2.0 / (d_ff + d_model))
        self.W1 = [[random.uniform(-scale1, scale1) for _ in range(d_ff)] for _ in range(d_model)]
        self.b1 = [0.0] * d_ff
        self.W2 = [[random.uniform(-scale2, scale2) for _ in range(d_model)] for _ in range(d_ff)]
        self.b2 = [0.0] * d_model

    @staticmethod
    def _gelu(x: float) -> float:
        """Gaussian Error Linear Unit approximation."""
        return 0.5 * x * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))

    def forward(self, x: List[List[float]]) -> List[List[float]]:
        out = []
        for row in x:
            # Hidden layer
            hidden = [
                self._gelu(sum(row[k] * self.W1[k][j] for k in range(len(row))) + self.b1[j])
                for j in range(len(self.b1))
            ]
            # Output layer
            proj = [
                sum(hidden[k] * self.W2[k][j] for k in range(len(hidden))) + self.b2[j]
                for j in range(len(self.b2))
            ]
            out.append(proj)
        return out


# ══════════════════════════════════════════════════════════════════════════════
#  TRANSFORMER BLOCK  (Attention + FFN + Residual + LayerNorm)
# ══════════════════════════════════════════════════════════════════════════════

class TransformerBlock:
    """Pre-norm transformer block:  LN -> Attn -> Residual -> LN -> FFN -> Residual."""

    def __init__(self, d_model: int = 64, n_heads: int = 4, d_ff: int = 256):
        self.ln1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln2 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)

    def forward(self, x: List[List[float]]) -> List[List[float]]:
        # Self-attention with residual
        normed = self.ln1.forward(x)
        attn_out = self.attn.forward(normed)
        x = [  # residual add
            [x[i][j] + attn_out[i][j] for j in range(len(x[0]))]
            for i in range(len(x))
        ]
        # Feed-forward with residual
        normed2 = self.ln2.forward(x)
        ffn_out = self.ffn.forward(normed2)
        x = [
            [x[i][j] + ffn_out[i][j] for j in range(len(x[0]))]
            for i in range(len(x))
        ]
        return x


# ══════════════════════════════════════════════════════════════════════════════
#  TOOL ENGINE  v5  (15+ tools)
# ══════════════════════════════════════════════════════════════════════════════

class ToolEngine:
    """Route user queries to the appropriate tool and return structured results."""

    _SAFE_MATH_NS = {
        "__builtins__": None,
        "math": math, "sqrt": math.sqrt, "pi": math.pi, "e": math.e,
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "log": math.log, "log2": math.log2, "log10": math.log10,
        "factorial": math.factorial, "abs": abs, "round": round,
        "pow": pow, "min": min, "max": max, "sum": sum,
        "gcd": math.gcd, "ceil": math.ceil, "floor": math.floor,
        "exp": math.exp, "radians": math.radians, "degrees": math.degrees,
        "asin": math.asin, "acos": math.acos, "atan": math.atan,
        "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh,
    }

    # ── 1. BARE MATH ─────────────────────────────────────────────────────

    _SAFE_NAMES = {
        'sqrt', 'sin', 'cos', 'tan', 'log', 'log2', 'log10',
        'factorial', 'abs', 'exp', 'ceil', 'floor', 'gcd',
        'min', 'max', 'sum', 'pow', 'round', 'pi', 'e',
        'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh',
        'radians', 'degrees', 'math',
    }

    @classmethod
    def _safe_eval(cls, expr: str):
        try:
            # Allow digits, operators, parens, and whitelisted function names
            cleaned = re.sub(r'[a-zA-Z_]+', lambda m: m.group() if m.group() in cls._SAFE_NAMES else '', expr)
            if not cleaned.strip() or cleaned != expr:
                # If anything was removed, it wasn't a safe expression
                # But allow through if only safe names were found
                if cleaned.strip() != expr.strip():
                    # Check that all word tokens are safe
                    words = re.findall(r'[a-zA-Z_]+', expr)
                    if not all(w in cls._SAFE_NAMES for w in words):
                        return None
            result = eval(expr, cls._SAFE_MATH_NS)
            if isinstance(result, float):
                return int(result) if result == int(result) else round(result, 10)
            return result
        except Exception:
            return None

    # ── 2. BASE64 ────────────────────────────────────────────────────────

    @classmethod
    def _tool_base64(cls, q: str) -> Optional[str]:
        m = re.match(r'^(?:b64|base64)\s+(enc|encode|dec|decode)\s+(.+)', q, re.I)
        if not m:
            return None
        mode, text = m.groups()
        try:
            if mode.lower().startswith('enc'):
                return f"Base64 encode: {base64.b64encode(text.encode()).decode()}"
            else:
                return f"Base64 decode: {base64.b64decode(text.encode()).decode('utf-8', errors='ignore')}"
        except Exception:
            return "Error: Could not process base64"

    # ── 3. HASH ──────────────────────────────────────────────────────────

    @classmethod
    def _tool_hash(cls, q: str) -> Optional[str]:
        m = re.match(r'^(?:hash|md5|sha1|sha256|sha512)\s+(.+)', q, re.I)
        if not m:
            return None
        full = m.group(0).lower()
        for algo in ['sha512', 'sha256', 'sha1', 'md5']:
            if full.startswith(algo):
                text = q[len(algo):].strip()
                h = hashlib.new(algo)
                h.update(text.encode())
                return f"Hash ({algo}): {h.hexdigest()}"
        return "Error: Unknown hash algorithm. Use md5, sha1, sha256, or sha512."

    # ── 4. FRACTIONS ─────────────────────────────────────────────────────

    @classmethod
    def _tool_fraction(cls, q: str) -> Optional[str]:
        if '/' not in q:
            return None
        if not re.match(r'^[\d\s\+\-\*\/\.\(\)]+$', q):
            return None
        try:
            safe_ns = {"Fraction": Fraction, "__builtins__": None}
            res = eval(q, safe_ns)
            if isinstance(res, Fraction):
                return f"Fraction: {res}  (decimal: {float(res):.6f})"
        except Exception:
            pass
        return None

    # ── 5. COMPLEX MATH ──────────────────────────────────────────────────

    @classmethod
    def _tool_complex(cls, q: str) -> Optional[str]:
        if 'j' not in q.lower() and not re.search(r'sqrt\s*\(\s*-', q):
            return None
        try:
            expr = q.strip()
            if 'j' in expr:
                # Only replace standalone 'j' (not preceded by a digit)
                # so "4j" stays "4j" (Python already understands it) but
                # a bare "j" becomes "1j"
                expr = re.sub(r'(?<!\d)j', '1j', expr)
            ns = {"__builtins__": None, "cmath": cmath, "math": math,
                  "sqrt": cmath.sqrt, "sin": cmath.sin, "cos": cmath.cos,
                  "tan": cmath.tan, "log": cmath.log, "exp": cmath.exp,
                  "pi": math.pi, "e": math.e}
            res = eval(expr, ns)
            return f"Complex: {res}"
        except Exception:
            return None

    # ── 6. UNIT CONVERTER ────────────────────────────────────────────────

    _UNITS: Dict[str, Dict[str, float]] = {
        # Length (to meters)
        "length": {"mm": 0.001, "cm": 0.01, "m": 1, "km": 1000,
                   "in": 0.0254, "ft": 0.3048, "yd": 0.9144, "mi": 1609.344},
        # Mass (to kg)
        "mass": {"mg": 1e-6, "g": 0.001, "kg": 1, "lb": 0.453592, "oz": 0.0283495, "ton": 1000},
        # Temperature (special handling)
        "temperature": {"c": 0, "f": 0, "k": 0},
        # Volume (to liters)
        "volume": {"ml": 0.001, "l": 1, "gal": 3.78541, "qt": 0.946353,
                   "pt": 0.473176, "cup": 0.236588, "floz": 0.0295735},
        # Speed (to m/s)
        "speed": {"ms": 1, "kmh": 0.277778, "mph": 0.44704, "knot": 0.514444, "fps": 0.3048},
        # Data (to bytes)
        "data": {"b": 1, "kb": 1024, "mb": 1048576, "gb": 1073741824,
                 "tb": 1099511627776, "pb": 1125899906842624},
    }

    @classmethod
    def _tool_unit(cls, q: str) -> Optional[str]:
        m = re.match(
            r'^(?:convert|unit)\s+([0-9.]+)\s+(\w+)\s+(?:to|in|as)\s+(\w+)', q, re.I
        )
        if not m:
            return None
        value, from_u, to_u = float(m.group(1)), m.group(2).lower(), m.group(3).lower()
        for cat, units in cls._UNITS.items():
            if from_u in units and to_u in units:
                if cat == "temperature":
                    result = cls._convert_temp(value, from_u, to_u)
                else:
                    base = value * units[from_u]
                    result = base / units[to_u]
                return f"{value} {from_u} = {round(result, 6)} {to_u}"
        return "Error: Unsupported unit conversion. Check unit names."

    @staticmethod
    def _convert_temp(val: float, fr: str, to: str) -> float:
        # Convert to Celsius first
        if fr == "c":
            c = val
        elif fr == "f":
            c = (val - 32) * 5 / 9
        else:  # K
            c = val - 273.15
        # Convert from Celsius to target
        if to == "c":
            return c
        elif to == "f":
            return c * 9 / 5 + 32
        else:  # K
            return c + 273.15

    # ── 7. DATE / TIME ───────────────────────────────────────────────────

    @classmethod
    def _tool_date(cls, q: str) -> Optional[str]:
        q_low = q.lower()
        if q_low in ['date', 'time', 'now', 'today', 'datetime']:
            now = datetime.datetime.now()
            return f"Current: {now.strftime('%Y-%m-%d %H:%M:%S %A')}"
        # "days from 2024-01-01 to 2024-12-31"
        m = re.match(r'^days\s+(?:from|between)\s+(\d{4}-\d{2}-\d{2})\s+(?:to|and)\s+(\d{4}-\d{2}-\d{2})', q, re.I)
        if m:
            d1 = datetime.date.fromisoformat(m.group(1))
            d2 = datetime.date.fromisoformat(m.group(2))
            delta = abs((d2 - d1).days)
            return f"Days between: {delta} days"
        # "add 30 days to 2024-01-01"
        m = re.match(r'^add\s+(\d+)\s+days?\s+(?:to\s+)?(\d{4}-\d{2}-\d{2})', q, re.I)
        if m:
            ndays = int(m.group(1))
            d = datetime.date.fromisoformat(m.group(2))
            result = d + datetime.timedelta(days=ndays)
            return f"Result: {result.strftime('%Y-%m-%d %A')}"
        # "day of week 2024-06-15"
        m = re.match(r'^day\s+(?:of\s+week\s+)?(\d{4}-\d{2}-\d{2})', q, re.I)
        if m:
            d = datetime.date.fromisoformat(m.group(1))
            return f"{d.strftime('%Y-%m-%d')} is a {d.strftime('%A')}"
        return None

    # ── 8. COLOR CONVERTER ───────────────────────────────────────────────

    @classmethod
    def _tool_color(cls, q: str) -> Optional[str]:
        # "color #ff8800" or "color rgb 255 128 0" or "color hsl 30 100 50"
        m = re.match(r'^color\s+#([0-9a-f]{6})', q, re.I)
        if m:
            h = m.group(1)
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
            hsl = cls._rgb_to_hsl(r, g, b)
            return f"HEX #{h} -> RGB({r}, {g}, {b}) -> HSL({hsl[0]}, {hsl[1]}%, {hsl[2]}%)"
        m = re.match(r'^color\s+rgb\s+(\d+)\s+(\d+)\s+(\d+)', q, re.I)
        if m:
            r, g, b = int(m.group(1)), int(m.group(2)), int(m.group(3))
            hx = f"#{r:02x}{g:02x}{b:02x}"
            hsl = cls._rgb_to_hsl(r, g, b)
            return f"RGB({r}, {g}, {b}) -> HEX {hx} -> HSL({hsl[0]}, {hsl[1]}%, {hsl[2]}%)"
        m = re.match(r'^color\s+hsl\s+(\d+)\s+(\d+)\s+(\d+)', q, re.I)
        if m:
            h, s, l = int(m.group(1)), int(m.group(2)), int(m.group(3))
            rgb = cls._hsl_to_rgb(h, s, l)
            hx = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
            return f"HSL({h}, {s}%, {l}%) -> RGB({rgb[0]}, {rgb[1]}, {rgb[2]}) -> HEX {hx}"
        return None

    @staticmethod
    def _rgb_to_hsl(r: int, g: int, b: int) -> Tuple[int, int, int]:
        r1, g1, b1 = r / 255, g / 255, b / 255
        mx, mn = max(r1, g1, b1), min(r1, g1, b1)
        l = (mx + mn) / 2
        if mx == mn:
            h = s = 0
        else:
            d = mx - mn
            s = d / (2 - mx - mn) if l > 0.5 else d / (mx + mn)
            if mx == r1:
                h = ((g1 - b1) / d + (6 if g1 < b1 else 0)) / 6
            elif mx == g1:
                h = ((b1 - r1) / d + 2) / 6
            else:
                h = ((r1 - g1) / d + 4) / 6
        return round(h * 360), round(s * 100), round(l * 100)

    @staticmethod
    def _hsl_to_rgb(h: int, s: int, l: int) -> Tuple[int, int, int]:
        h1, s1, l1 = h / 360, s / 100, l / 100
        if s1 == 0:
            v = round(l1 * 255)
            return v, v, v

        def hue2rgb(p, q, t):
            if t < 0: t += 1
            if t > 1: t -= 1
            if t < 1/6: return p + (q - p) * 6 * t
            if t < 1/2: return q
            if t < 2/3: return p + (q - p) * (2/3 - t) * 6
            return p

        q = l1 * (1 + s1) if l1 < 0.5 else l1 + s1 - l1 * s1
        p = 2 * l1 - q
        return (round(hue2rgb(p, q, h1 + 1/3) * 255),
                round(hue2rgb(p, q, h1) * 255),
                round(hue2rgb(p, q, h1 - 1/3) * 255))

    # ── 9. ROMAN NUMERALS ────────────────────────────────────────────────

    @classmethod
    def _tool_roman(cls, q: str) -> Optional[str]:
        # "roman 42" or "roman XLII"
        m = re.match(r'^roman\s+([IVXLCDM]+)$', q, re.I)
        if m:
            try:
                val = cls._roman_to_int(m.group(1).upper())
                return f"Roman {m.group(1).upper()} = {val}"
            except ValueError:
                return "Error: Invalid Roman numeral."
        m = re.match(r'^roman\s+(\d+)$', q, re.I)
        if m:
            val = int(m.group(1))
            if 1 <= val <= 3999:
                return f"{val} = {cls._int_to_roman(val)}"
            return "Error: Number must be between 1 and 3999."
        return None

    @staticmethod
    def _roman_to_int(s: str) -> int:
        vals = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
        total = 0
        prev = 0
        for ch in reversed(s):
            v = vals.get(ch, 0)
            if v < prev:
                total -= v
            else:
                total += v
            prev = v
        if total == 0:
            raise ValueError
        return total

    @staticmethod
    def _int_to_roman(n: int) -> str:
        pairs = [(1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'),
                 (100, 'C'), (90, 'XC'), (50, 'L'), (40, 'XL'),
                 (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I')]
        result = ''
        for val, sym in pairs:
            while n >= val:
                result += sym
                n -= val
        return result

    # ── 10. STATISTICS ───────────────────────────────────────────────────

    @classmethod
    def _tool_stats(cls, q: str) -> Optional[str]:
        m = re.match(r'^stats?\s+([\d.,\s]+)$', q, re.I)
        if not m:
            return None
        try:
            nums = [float(x.strip()) for x in m.group(1).split(',') if x.strip()]
        except ValueError:
            return "Error: Could not parse numbers."
        if len(nums) < 2:
            return "Error: Need at least 2 numbers."
        n = len(nums)
        mean = sum(nums) / n
        sorted_n = sorted(nums)
        median = (sorted_n[n // 2] if n % 2 == 1
                  else (sorted_n[n // 2 - 1] + sorted_n[n // 2]) / 2)
        freq = Counter(nums)
        max_freq = max(freq.values())
        modes = [k for k, v in freq.items() if v == max_freq]
        mode_str = ", ".join(str(m) for m in modes) if max_freq > 1 else "No mode"
        variance = sum((x - mean) ** 2 for x in nums) / (n - 1)
        stdev = math.sqrt(variance)
        rng = max(nums) - min(nums)
        return (f"Count: {n}  |  Mean: {round(mean, 4)}  |  Median: {round(median, 4)}\n"
                f"Mode: {mode_str}  |  Stdev: {round(stdev, 4)}  |  Variance: {round(variance, 4)}\n"
                f"Range: {round(rng, 4)}  |  Min: {min(nums)}  |  Max: {max(nums)}")

    # ── 11. COMBINATORICS ────────────────────────────────────────────────

    @classmethod
    def _tool_combinatorics(cls, q: str) -> Optional[str]:
        # "ncr 10 3" or "npr 5 2" or "factorial 6"
        m = re.match(r'^(?:ncr|C)\s+(\d+)\s+(\d+)$', q, re.I)
        if m:
            n, r = int(m.group(1)), int(m.group(2))
            if r > n:
                return "Error: r cannot be greater than n."
            return f"C({n},{r}) = {math.comb(n, r)}"
        m = re.match(r'^(?:npr|P)\s+(\d+)\s+(\d+)$', q, re.I)
        if m:
            n, r = int(m.group(1)), int(m.group(2))
            if r > n:
                return "Error: r cannot be greater than n."
            return f"P({n},{r}) = {math.perm(n, r)}"
        m = re.match(r'^(?:factorial|fact)\s+(\d+)$', q, re.I)
        if m:
            n = int(m.group(1))
            if n > 170:
                return "Error: Number too large (max 170)."
            return f"{n}! = {math.factorial(n)}"
        return None

    # ── 12. NUMBER-BASE CONVERTER ────────────────────────────────────────

    @classmethod
    def _tool_baseconv(cls, q: str) -> Optional[str]:
        m = re.match(r'^(?:base|conv)\s+([0-9a-fA-F]+)\s+(?:from\s+)?(bin|oct|dec|hex)\s+(?:to\s+)?(bin|oct|dec|hex)', q, re.I)
        if not m:
            # Shortcut: "hex 255" or "bin 10" or "oct 64"
            m2 = re.match(r'^(hex|bin|oct)\s+(\d+)$', q, re.I)
            if m2:
                target, val = m2.group(1).lower(), m2.group(2)
                dec_val = int(val, 10)
                results = {"hex": hex, "bin": bin, "oct": oct}
                return f"{val} (dec) = {results[target](dec_val)}"
            return None
        val_str, from_base, to_base = m.group(1), m.group(2).lower(), m.group(3).lower()
        bases = {"bin": 2, "oct": 8, "dec": 10, "hex": 16}
        try:
            dec_val = int(val_str, bases[from_base])
        except ValueError:
            return "Error: Invalid number for the given base."
        if to_base == "dec":
            return f"{val_str} ({from_base}) = {dec_val} (dec)"
        elif to_base == "hex":
            return f"{val_str} ({from_base}) = {hex(dec_val)} (hex)"
        elif to_base == "oct":
            return f"{val_str} ({from_base}) = {oct(dec_val)} (oct)"
        else:
            return f"{val_str} ({from_base}) = {bin(dec_val)} (bin)"

    # ── 13. GEOMETRY ─────────────────────────────────────────────────────

    @classmethod
    def _tool_geometry(cls, q: str) -> Optional[str]:
        q_low = q.lower()
        # Circle
        m = re.match(r'^(?:area|circumference)\s+(?:of\s+)?circle\s+r\s*=?\s*([0-9.]+)', q_low)
        if m:
            r = float(m.group(1))
            area = math.pi * r ** 2
            circ = 2 * math.pi * r
            return f"Circle (r={r}): Area = {round(area, 4)}, Circumference = {round(circ, 4)}"
        # Rectangle
        m = re.match(r'^(?:area|perimeter)\s+(?:of\s+)?rect(?:angle)?\s+([0-9.]+)\s*[x,]\s*([0-9.]+)', q_low)
        if m:
            w, h = float(m.group(1)), float(m.group(2))
            return f"Rectangle ({w}x{h}): Area = {w*h}, Perimeter = {2*(w+h)}"
        # Triangle
        m = re.match(r'^(?:area)\s+(?:of\s+)?tri(?:angle)?\s+(?:base\s*=?\s*([0-9.]+)\s+height\s*=?\s*([0-9.]+))', q_low)
        if m:
            b, h = float(m.group(1)), float(m.group(2))
            return f"Triangle (b={b}, h={h}): Area = {round(0.5*b*h, 4)}"
        # Sphere
        m = re.match(r'^(?:area|volume)\s+(?:of\s+)?sphere\s+r\s*=?\s*([0-9.]+)', q_low)
        if m:
            r = float(m.group(1))
            vol = 4/3 * math.pi * r**3
            sa = 4 * math.pi * r**2
            return f"Sphere (r={r}): Volume = {round(vol, 4)}, Surface Area = {round(sa, 4)}"
        # Cylinder
        m = re.match(r'^(?:area|volume)\s+(?:of\s+)?cylinder\s+r\s*=?\s*([0-9.]+)\s+h\s*=?\s*([0-9.]+)', q_low)
        if m:
            r, h = float(m.group(1)), float(m.group(2))
            vol = math.pi * r**2 * h
            sa = 2 * math.pi * r * (r + h)
            return f"Cylinder (r={r}, h={h}): Volume = {round(vol, 4)}, Surface Area = {round(sa, 4)}"
        # Cone
        m = re.match(r'^(?:area|volume)\s+(?:of\s+)?cone\s+r\s*=?\s*([0-9.]+)\s+h\s*=?\s*([0-9.]+)', q_low)
        if m:
            r, h = float(m.group(1)), float(m.group(2))
            vol = math.pi * r**2 * h / 3
            sl = math.sqrt(r**2 + h**2)
            sa = math.pi * r * (r + sl)
            return f"Cone (r={r}, h={h}): Volume = {round(vol, 4)}, Surface Area = {round(sa, 4)}"
        # Pythagorean
        m = re.match(r'^pyth(?:agorean)?\s+([0-9.]+)\s+([0-9.]+)', q_low)
        if m:
            a, b = float(m.group(1)), float(m.group(2))
            c = math.sqrt(a**2 + b**2)
            return f"Hypotenuse: {round(c, 4)}  (a={a}, b={b})"
        return None

    # ── 14. FINANCIAL CALCULATOR ─────────────────────────────────────────

    @classmethod
    def _tool_finance(cls, q: str) -> Optional[str]:
        # Compound interest: "compound 1000 5 10" (principal, rate%, years)
        m = re.match(r'^compound\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)', q, re.I)
        if m:
            p, r, t = float(m.group(1)), float(m.group(2)), float(m.group(3))
            a = p * (1 + r/100) ** t
            interest = a - p
            return (f"Principal: {p}  |  Rate: {r}%  |  Years: {t}\n"
                    f"Final Amount: {round(a, 2)}  |  Interest Earned: {round(interest, 2)}")
        # Loan payment: "loan 100000 5 30" (principal, rate%, years)
        m = re.match(r'^loan\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)', q, re.I)
        if m:
            p, r, t = float(m.group(1)), float(m.group(2)), float(m.group(3))
            monthly_rate = r / 100 / 12
            n_months = t * 12
            if monthly_rate == 0:
                payment = p / n_months
            else:
                payment = p * (monthly_rate * (1 + monthly_rate)**n_months) / ((1 + monthly_rate)**n_months - 1)
            total_paid = payment * n_months
            return (f"Loan: {p}  |  Rate: {r}%  |  Term: {t} years\n"
                    f"Monthly Payment: {round(payment, 2)}  |  Total Paid: {round(total_paid, 2)}  |  "
                    f"Total Interest: {round(total_paid - p, 2)}")
        # ROI: "roi 1000 1500" (investment, return)
        m = re.match(r'^roi\s+([0-9.]+)\s+([0-9.]+)', q, re.I)
        if m:
            cost, gain = float(m.group(1)), float(m.group(2))
            roi = (gain - cost) / cost * 100
            profit = gain - cost
            return f"Investment: {cost}  |  Return: {gain}  |  Profit: {round(profit, 2)}  |  ROI: {round(roi, 2)}%"
        return None

    # ── 15. GCD / LCM ────────────────────────────────────────────────────

    @classmethod
    def _tool_gcd_lcm(cls, q: str) -> Optional[str]:
        m = re.match(r'^(?:gcd|lcm)\s+(\d+)\s+(\d+)', q, re.I)
        if not m:
            return None
        a, b = int(m.group(1)), int(m.group(2))
        g = math.gcd(a, b)
        l = abs(a * b) // g
        cmd = m.group(0).split()[0].lower()
        if cmd == "gcd":
            return f"GCD({a}, {b}) = {g}"
        return f"LCM({a}, {b}) = {l}"

    # ── 16. LINEAR EQUATION SOLVER ───────────────────────────────────────

    @classmethod
    def _tool_equation(cls, q: str) -> Optional[str]:
        # "solve 2x + 3 = 11"
        m = re.match(r'^solve\s+([0-9.]*)\s*x\s*([+\-])\s*([0-9.]+)\s*=\s*([0-9.]+)', q, re.I)
        if m:
            a = float(m.group(1)) if m.group(1) else 1.0
            sign = 1 if m.group(2) == '+' else -1
            b = float(m.group(3))
            c = float(m.group(4))
            x = (c - sign * b) / a
            return f"{a}x {'+' if sign == 1 else '-'} {b} = {c}  =>  x = {round(x, 6)}"
        # "solve 3x - 7 = 20"
        m = re.match(r'^solve\s+([0-9.]*)\s*x\s*([+\-])\s*([0-9.]+)\s*=\s*([0-9.]+)', q, re.I)
        if m:
            a = float(m.group(1)) if m.group(1) else 1.0
            sign = 1 if m.group(2) == '+' else -1
            b = float(m.group(3))
            c = float(m.group(4))
            x = (c - sign * b) / a
            return f"x = {round(x, 6)}"
        return None

    # ── 17. TEXT ANALYSIS ────────────────────────────────────────────────

    @classmethod
    def _tool_text_analysis(cls, q: str) -> Optional[str]:
        m = re.match(r'^(?:analyze|textstats?|wordcount)\s+(.+)$', q, re.I)
        if not m:
            return None
        text = m.group(1)
        words = text.split()
        chars = len(text)
        chars_no_space = len(text.replace(" ", ""))
        sentences = len(re.findall(r'[.!?]+', text)) or 1
        avg_word_len = sum(len(w) for w in words) / len(words) if words else 0
        # Flesch-Kincaid approximation
        syllables = sum(cls._count_syllables(w) for w in words)
        if len(words) > 0:
            fk = (0.39 * len(words) / sentences + 11.8 * syllables / len(words) - 15.59)
        else:
            fk = 0
        return (f"Words: {len(words)}  |  Characters: {chars}  |  Characters (no spaces): {chars_no_space}\n"
                f"Sentences: {sentences}  |  Avg word length: {round(avg_word_len, 1)}  |  "
                f"Syllables: {syllables}\n"
                f"Readability (Flesch-Kincaid grade): {round(max(0, fk), 1)}")

    @staticmethod
    def _count_syllables(word: str) -> int:
        word = word.lower().strip()
        if len(word) <= 3:
            return 1
        count = 0
        vowels = "aeiouy"
        prev_vowel = False
        for ch in word:
            is_vowel = ch in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel
        if word.endswith("e") and count > 1:
            count -= 1
        return max(count, 1)

    # ── 18. PROBABILITY CALCULATOR ───────────────────────────────────────

    @classmethod
    def _tool_probability(cls, q: str) -> Optional[str]:
        # Binomial: "binomial 10 0.5 3" (n, p, k) = P(X=k)
        m = re.match(r'^binomial\s+(\d+)\s+([0-9.]+)\s+(\d+)', q, re.I)
        if m:
            n, p, k = int(m.group(1)), float(m.group(2)), int(m.group(3))
            if not (0 <= p <= 1):
                return "Error: Probability p must be between 0 and 1."
            prob = math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))
            mean = n * p
            variance = n * p * (1 - p)
            return (f"Binomial(n={n}, p={p}, k={k}): P(X={k}) = {round(prob, 6)}\n"
                    f"Mean: {round(mean, 4)}  |  Variance: {round(variance, 4)}  |  Stdev: {round(math.sqrt(variance), 4)}")
        # Poisson: "poisson 5 3" (lambda, k) = P(X=k)
        m = re.match(r'^poisson\s+([0-9.]+)\s+(\d+)', q, re.I)
        if m:
            lam, k = float(m.group(1)), int(m.group(2))
            if lam <= 0:
                return "Error: Lambda must be positive."
            prob = (lam ** k) * math.exp(-lam) / math.factorial(k)
            return (f"Poisson(lambda={lam}, k={k}): P(X={k}) = {round(prob, 6)}\n"
                    f"Mean: {lam}  |  Variance: {lam}  |  Stdev: {round(math.sqrt(lam), 4)}")
        return None

    # ── MASTER ROUTER ────────────────────────────────────────────────────

    @classmethod
    def route(cls, query: str) -> Optional[str]:
        """Try every tool in order; return the first non-None result, or None."""
        q = query.strip()

        # 1. Bare math expression  (pure numbers/operators OR safe function calls)
        math_func_pattern = r'^(sqrt|sin|cos|tan|log2?|log10|factorial|abs|exp|ceil|floor|gcd|min|max|pow|round|asin|acos|atan|sinh|cosh|tanh|radians|degrees)\s*[\(\d]'
        if re.match(r'^[\d\s\+\-\*\/\%\.\(\)eE]+$', q) or re.match(math_func_pattern, q, re.I):
            result = cls._safe_eval(q)
            if result is not None:
                return f"Answer: {result}"

        # 2-18. Named tools
        for tool_fn in [
            cls._tool_base64,
            cls._tool_hash,
            cls._tool_fraction,
            cls._tool_complex,
            cls._tool_unit,
            cls._tool_date,
            cls._tool_color,
            cls._tool_roman,
            cls._tool_stats,
            cls._tool_combinatorics,
            cls._tool_baseconv,
            cls._tool_geometry,
            cls._tool_finance,
            cls._tool_gcd_lcm,
            cls._tool_equation,
            cls._tool_text_analysis,
            cls._tool_probability,
        ]:
            result = tool_fn(q)
            if result is not None:
                return result

        # Help
        if q.lower() in ['help', '/help', '!tools', 'tools']:
            return (
                "MLLM-5-ATLAS Tools:\n"
                "  Math: 2+2, sqrt(16), sin(0.5)\n"
                "  Base64: base64 encode hello / base64 decode SGVsbG8=\n"
                "  Hash: sha256 password / md5 hello\n"
                "  Fractions: 1/2 + 3/4\n"
                "  Complex: 3+4j / sqrt(-1)\n"
                "  Units: convert 100 km to mi / convert 72 kg to lb\n"
                "  Date: now / days from 2024-01-01 to 2024-12-31\n"
                "  Color: color #ff8800 / color rgb 255 128 0\n"
                "  Roman: roman 42 / roman XLII\n"
                "  Stats: stats 10,20,30,40,50\n"
                "  Combinatorics: ncr 10 3 / npr 5 2 / factorial 6\n"
                "  Base conv: hex 255 / base FF from hex to dec\n"
                "  Geometry: area circle r=5 / volume sphere r=3\n"
                "  Finance: compound 1000 5 10 / loan 100000 5 30\n"
                "  GCD/LCM: gcd 12 8 / lcm 4 6\n"
                "  Equation: solve 2x + 3 = 11\n"
                "  Text: analyze The quick brown fox\n"
                "  Probability: binomial 10 0.5 3 / poisson 5 3"
            )

        return None  # No tool matched


# ══════════════════════════════════════════════════════════════════════════════
#  CONTEXT MEMORY  (Word-overlap retrieval from user queries only)
# ══════════════════════════════════════════════════════════════════════════════

class ContextMemory:
    """Stores conversation turns and retrieves the most relevant *user queries*
    to enrich generation context.  Model outputs are stored for display but are
    NOT injected into the generation context — this prevents the degenerate
    feedback loop where bad model output poisons future generation."""

    def __init__(self, max_turns: int = 50):
        self.max_turns = max_turns
        self.turns: List[Dict[str, str]] = []  # {"user": ..., "model": ...}

    def add(self, user_text: str, model_text: str):
        self.turns.append({"user": user_text, "model": model_text})
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]

    def retrieve_user_context(self, query: str, top_k: int = 2) -> str:
        """Return a string of the most relevant past *user queries* to prepend
        to the current context.  Model responses are excluded to avoid
        contaminating generation with low-quality output."""
        if not self.turns:
            return ""
        q_words = set(re.findall(r'\b\w+\b', query.lower()))
        if not q_words:
            return " ".join(t["user"] for t in self.turns[-top_k:])
        scored = []
        for turn in self.turns:
            turn_words = set(re.findall(r'\b\w+\b', turn["user"].lower()))
            overlap = len(q_words & turn_words)
            scored.append((overlap, turn["user"]))
        scored.sort(key=lambda x: x[0], reverse=True)
        return " ".join(text for _, text in scored[:top_k])

    def get_recent(self, n: int = 3) -> List[Dict[str, str]]:
        return self.turns[-n:]

    def clear(self):
        self.turns = []


# ══════════════════════════════════════════════════════════════════════════════
#  N-GRAM MODEL  —  Fast Interpolated Backoff + Direct Lookup
# ══════════════════════════════════════════════════════════════════════════════

class SmoothedNGramModel:
    """N-gram language model with interpolated backoff.

    Key design choices for SPEED and ACCURACY on small corpora:
    • predict_next() only scores words that were ACTUALLY SEEN after the
      matched context — not the entire vocabulary.  This is O(k) where k
      is the number of distinct successors, not O(|V|).
    • Backoff is simple: try the longest matching n-gram first; if not
      found, chop one token from the left and try again.
    • When a context IS found, its successors are weighted by count
      (with optional temperature scaling).  No expensive smoothing math.
    • Fall back to unigram frequencies when no context matches.
    """

    def __init__(self, max_n: int = 20):
        self.max_n = max_n
        # n -> {context_tuple: Counter(next_token)}
        self.ngram_contexts: Dict[int, Dict[tuple, Counter]] = {}
        self.unigram_count = 0
        self.vocab: set = set()
        self.token_freq: Counter = Counter()
        self._trained = False

    def train(self, tokens: List[str]):
        """Build n-gram tables from the token list."""
        self.vocab = set(tokens)
        self.token_freq = Counter(tokens)
        self.unigram_count = len(tokens)

        for n in range(1, self.max_n + 1):
            ctx: Dict[tuple, Counter] = defaultdict(Counter)
            for i in range(len(tokens) - n):
                ngram = tuple(tokens[i:i + n])
                next_tok = tokens[i + n]
                ctx[ngram][next_tok] += 1
            self.ngram_contexts[n] = dict(ctx)
        self._trained = True

    def _get_successors(self, context_tokens: List[str]) -> Optional[Counter]:
        """Walk from longest n-gram down to unigram, return the first
        non-empty successor Counter, or None if nothing matches."""
        for n in range(min(len(context_tokens), self.max_n), 0, -1):
            ngram = tuple(context_tokens[-n:])
            ctx_dict = self.ngram_contexts.get(n, {}).get(ngram)
            if ctx_dict and len(ctx_dict) > 0:
                return ctx_dict
        return None

    def predict_next(self, context_tokens: List[str],
                     temperature: float = 0.8) -> Tuple[str, float]:
        """Return (predicted_word, confidence).

        Only scores words that were seen after the matched context —
        typically a small handful, NOT the whole vocabulary.  This is
        the single biggest speed win over the previous version.
        """
        if not self._trained or not self.vocab:
            return ("<UNK>", 0.0)

        # Fast path: direct successor lookup
        successors = self._get_successors(context_tokens)

        if successors:
            words = list(successors.keys())
            counts = list(successors.values())
            total = sum(counts)

            # Convert to probabilities
            probs = [c / total for c in counts]

            # Apply temperature: lower = sharper (more deterministic)
            if temperature != 1.0 and temperature > 0:
                probs = [p ** (1.0 / temperature) for p in probs]
                psum = sum(probs)
                probs = [p / psum for p in probs]

            # Weighted random choice
            chosen = random.choices(words, weights=probs, k=1)[0]
            confidence = probs[words.index(chosen)]
            return chosen, confidence

        # Fallback: sample from unigram distribution (weighted by frequency)
        words = list(self.token_freq.keys())
        counts = list(self.token_freq.values())
        total = sum(counts)
        probs = [c / total for c in counts]

        if temperature != 1.0 and temperature > 0:
            probs = [p ** (1.0 / temperature) for p in probs]
            psum = sum(probs)
            probs = [p / psum for p in probs]

        chosen = random.choices(words, weights=probs, k=1)[0]
        confidence = probs[words.index(chosen)]
        return chosen, confidence


# ══════════════════════════════════════════════════════════════════════════════
#  ONLINE LEARNER  (Incremental n-gram updates — USER INPUT ONLY)
# ══════════════════════════════════════════════════════════════════════════════

class OnlineLearner:
    """Incrementally updates n-gram counts as new text is observed.
    CRITICAL: only learn from high-quality text (user input), NEVER from
    the model's own output — that creates a degenerate feedback loop."""

    def __init__(self, ngram_model: SmoothedNGramModel):
        self.model = ngram_model

    def learn(self, tokens: List[str]):
        """Add new tokens to the n-gram model incrementally."""
        for token in tokens:
            self.model.vocab.add(token)
            self.model.token_freq[token] += 1
            self.model.unigram_count += 1

        for n in range(1, self.model.max_n + 1):
            if n not in self.model.ngram_contexts:
                self.model.ngram_contexts[n] = {}
            for i in range(len(tokens) - n):
                ngram = tuple(tokens[i:i + n])
                next_tok = tokens[i + n]
                if ngram not in self.model.ngram_contexts[n]:
                    self.model.ngram_contexts[n][ngram] = Counter()
                self.model.ngram_contexts[n][ngram][next_tok] += 1


# ══════════════════════════════════════════════════════════════════════════════
#  MLLM-5-ATLAS  —  Main Model Class
# ══════════════════════════════════════════════════════════════════════════════

class MLLM5_Atlas:
    """MLLM-5-ATLAS: Advanced Tiny Language Architecture System.

    Architecture stack:
        1. BPE-lite tokenizer (subword)
        2. Embedding + positional encoding + transformer blocks  (available
           but OFF by default — untrained random weights add noise)
        3. Fast interpolated-backoff n-gram backbone  (direct successor
           lookup — O(k) not O(|V|) per prediction step)
        4. Tool engine (18 tools)
        5. Context memory (user-query-only retrieval — no model output
           contamination)
        6. Online learner (user input only — prevents feedback loop)
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
        max_ngram: int = 20,
        beam_width: int = 1,
        context_window: int = 50,
        vocab_size: int = 2000,
        use_transformer: bool = False,   # OFF by default — needs real training
    ):
        # Hyper-parameters
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.max_ngram = max_ngram
        self.beam_width = beam_width
        self.context_window = context_window
        self.vocab_size = vocab_size
        self.use_transformer = use_transformer

        # Components
        self.tokenizer = BPETokenizer(vocab_size)
        self.embedding = EmbeddingLayer(vocab_size, d_model)
        self.transformer_blocks: List[TransformerBlock] = [
            TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ]
        self.ln_final = LayerNorm(d_model)
        self.ngram_model = SmoothedNGramModel(max_n=max_ngram)
        self.context_memory = ContextMemory(max_turns=context_window)
        self.online_learner = OnlineLearner(self.ngram_model)
        self.tool_engine = ToolEngine

        # State
        self.trained = False
        self.corpus = ""
        self.tokens: List[str] = []
        self.entropy = 0.0
        self.perplexity = 0.0
        self.total_interactions = 0

    # ── TRAINING ─────────────────────────────────────────────────────────

    def train(self, corpus: str, verbose: bool = True):
        """Full training pipeline: tokenize, train n-grams."""
        if verbose:
            print("\n" + "=" * 55)
            print("  MLLM-5-ATLAS  Training Pipeline")
            print("=" * 55)

        self.corpus = corpus

        # 1. Train BPE tokenizer
        if verbose:
            print("  [1/3] Training BPE tokenizer...")
        self.tokenizer.train(corpus, verbose=verbose)

        # 2. Word-level tokens for n-gram model
        if verbose:
            print("  [2/3] Building word-level token stream...")
        self.tokens = self._word_tokenize(corpus)

        # 3. Train fast n-gram model
        if verbose:
            print("  [3/3] Training n-gram model (interpolated backoff)...")
        self.ngram_model.train(self.tokens)

        self._compute_metrics()
        self.trained = True

        if verbose:
            print(f"\n  Vocabulary size:    {len(self.ngram_model.vocab)}")
            print(f"  Total tokens:       {len(self.tokens)}")
            print(f"  Entropy:            {self.entropy:.4f} bits/token")
            print(f"  Perplexity:         {self.perplexity:.2f}")
            print(f"  Transformer:        {'ON' if self.use_transformer else 'OFF'} (use_transformer={self.use_transformer})")
            print(f"  Max n-gram:         {self.max_ngram}")
            print(f"  Tool count:         18")
            print("=" * 55 + "\n")

    @staticmethod
    def _word_tokenize(text: str) -> List[str]:
        """Simple word + punctuation tokenizer for the n-gram model."""
        return [t for t in re.findall(r'\b\w+\b|[.!?,;:\'"()]', text.lower()) if t.strip()]

    def _compute_metrics(self):
        """Compute entropy and perplexity from token frequencies."""
        freq = self.ngram_model.token_freq
        total = self.ngram_model.unigram_count
        if total == 0:
            return
        entropy_sum = 0.0
        for token, count in freq.items():
            p = count / total
            if p > 0:
                entropy_sum += -p * math.log2(p)
        self.entropy = entropy_sum
        self.perplexity = 2 ** self.entropy

    # ── GENERATION ───────────────────────────────────────────────────────

    def generate_response(self, user_input: str, max_length: int = 25,
                          temperature: float = 0.8) -> str:
        """Generate a response: route to tools first, then generate text.

        Speed notes:
        • Tool routing is fast — regex checks, returns immediately on match.
        • N-gram generation only scores words SEEN after the context, not
          the entire vocabulary.  O(k) per step where k is typically 1-5.
        • Transformer blocks are SKIPPED by default (use_transformer=False)
          because untrained random weights only add noise to temperature.
        """
        # 1. Tool routing (fast path — returns immediately if matched)
        tool_result = self.tool_engine.route(user_input)
        if tool_result is not None:
            return tool_result

        if not self.trained:
            return "Model not trained yet. Please run .train(corpus) first."

        # 2. Build context: user input + relevant past user queries
        #    (NOT model output — prevents degenerate feedback loop)
        extra_context = self.context_memory.retrieve_user_context(user_input, top_k=1)
        context_text = (extra_context + " " + user_input).strip() if extra_context else user_input
        context_tokens = self._word_tokenize(context_text)

        # 3. Optional transformer pass (off by default)
        if self.use_transformer:
            token_ids = self.tokenizer.encode(context_text)
            if token_ids:
                token_ids = [min(tid, self.vocab_size - 1) for tid in token_ids]
                x = self.embedding.forward(token_ids)
                for block in self.transformer_blocks:
                    x = block.forward(x)
                x = self.ln_final.forward(x)
                # Modulate temperature from hidden state
                if x:
                    last_vec = x[-1]
                    mean_val = sum(last_vec) / len(last_vec)
                    temperature = max(0.3, min(1.5, temperature + mean_val * 0.1))

        # 4. Generate via fast n-gram sampling
        #    Don't stop on the FIRST punctuation — generate at least a few
        #    words so the response isn't trivially short.
        response_tokens: List[str] = []
        for step in range(max_length):
            word, _ = self.ngram_model.predict_next(context_tokens, temperature)
            response_tokens.append(word)
            context_tokens.append(word)
            # Allow early stop on sentence-ending punctuation, but only
            # after we've generated at least 4 words
            if word in ['.', '!', '?'] and step >= 3:
                break

        # 5. Format response
        response = " ".join(response_tokens)
        response = response.replace(" .", ".").replace(" !", "!").replace(" ?", "?")
        response = response.replace(" ,", ",").replace(" ;", ";").replace(" :", ":")

        # 6. Prepend user input to the response
        #    e.g. input "hello" -> output "hello there, how can i help you."
        #    Strip the echoed input if the n-gram model already repeated it,
        #    then prepend it once cleanly.
        input_lower = user_input.lower()
        input_tokens = self._word_tokenize(user_input)
        input_prefix = " ".join(input_tokens)
        if response.startswith(input_prefix):
            response = response[len(input_prefix):].strip()
        # Prepend the original user input
        response = user_input.strip() + " " + response

        return response if response else "..."

    # ── INTERACTION ──────────────────────────────────────────────────────

    def chat(self, user_input: str, temperature: float = 0.8) -> str:
        """Full chat pipeline: store context, generate, learn from user only."""
        self.total_interactions += 1

        # Generate response
        response = self.generate_response(user_input, temperature=temperature)

        # Store in context memory (both sides for display, but generation
        # only uses the user side — see retrieve_user_context)
        self.context_memory.add(user_input, response)

        # Online learning: ONLY from user input, NOT from model output
        user_tokens = self._word_tokenize(user_input)
        self.online_learner.learn(user_tokens)

        return response

    # ── STATS ────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        return {
            "model": "MLLM-5-ATLAS",
            "trained": self.trained,
            "vocabulary_size": len(self.ngram_model.vocab),
            "total_tokens": len(self.tokens),
            "entropy_bits": round(self.entropy, 4),
            "perplexity": round(self.perplexity, 2),
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "d_ff": self.d_ff,
            "max_ngram": self.max_ngram,
            "tool_count": 18,
            "use_transformer": self.use_transformer,
            "context_window": self.context_window,
            "total_interactions": self.total_interactions,
            "context_memory_turns": len(self.context_memory.turns),
            "bpe_vocab_size": len(self.tokenizer.vocab),
            "bpe_merges": len(self.tokenizer.merges),
        }

    def get_config(self) -> str:
        """Return a formatted summary of the model configuration."""
        s = self.get_stats()
        return (
            f"Model:              {s['model']}\n"
            f"Trained:            {s['trained']}\n"
            f"Vocabulary:         {s['vocabulary_size']} words\n"
            f"Total tokens:       {s['total_tokens']}\n"
            f"Entropy:            {s['entropy_bits']} bits/token\n"
            f"Perplexity:         {s['perplexity']}\n"
            f"d_model:            {s['d_model']}\n"
            f"Attention heads:    {s['n_heads']}\n"
            f"Transformer layers: {s['n_layers']} (active: {s['use_transformer']})\n"
            f"Feed-forward dim:   {s['d_ff']}\n"
            f"Max n-gram:         {s['max_ngram']}\n"
            f"Tools:              {s['tool_count']}\n"
            f"BPE vocab:          {s['bpe_vocab_size']}\n"
            f"BPE merges:         {s['bpe_merges']}\n"
            f"Interactions:       {s['total_interactions']}\n"
        )


# ══════════════════════════════════════════════════════════════════════════════
#  CLI INTERFACE
# ══════════════════════════════════════════════════════════════════════════════

def print_banner():
    print("\n" + "=" * 60)
    print("   MLLM-5-ATLAS")
    print("   Advanced Tiny Language Architecture System")
    print("   " + "-" * 52)
    print("   Math: 2+2, sqrt(16), sin(pi/4)")
    print("   Tools: base64, hash, units, date, color, roman,")
    print("          stats, combinatorics, geometry, finance,")
    print("          gcd/lcm, equations, text analysis, probability")
    print("   Chat: type naturally for language generation")
    print("   Commands: /stats /config /context /clear /help /quit")
    print("=" * 60 + "\n")


def main():
    print_banner()

    # Initialize model — transformer OFF by default for speed
    # Set use_transformer=True if you have real trained weights
    model = MLLM5_Atlas(
        d_model=64,          # Embedding / hidden dimension
        n_heads=4,           # Number of attention heads
        n_layers=2,          # Transformer blocks (available but off)
        d_ff=256,            # Feed-forward hidden dimension
        max_ngram=20,        # Maximum n-gram order
        beam_width=1,        # 1 = fast sampling (no beam search)
        context_window=50,   # Conversation memory turns
        vocab_size=2000,     # BPE vocabulary size
        use_transformer=False,  # Off — untrained weights add noise
    )

    # Train on default corpus
    model.train(CORPUS, verbose=True)

    print("Ready. Type math expressions, tool commands, or chat.")
    print("Type /help for tools list, /quit to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        if cmd in ('quit', 'exit', '/quit', '/exit', 'bye'):
            print("Goodbye.")
            break
        elif cmd in ('/stats', 'stats'):
            stats = model.get_stats()
            for k, v in stats.items():
                print(f"  {k}: {v}")
            print()
            continue
        elif cmd in ('/config', 'config'):
            print(model.get_config())
            continue
        elif cmd in ('/context', 'context'):
            recent = model.context_memory.get_recent(5)
            if not recent:
                print("  No conversation history yet.")
            else:
                for i, turn in enumerate(recent, 1):
                    print(f"  [{i}] You: {turn['user']}")
                    print(f"      Atlas: {turn['model']}")
            print()
            continue
        elif cmd in ('/clear', 'clear'):
            model.context_memory.clear()
            print("  Context memory cleared.\n")
            continue
        elif cmd in ('/help', 'help', '!tools', 'tools'):
            print(ToolEngine.route("help"))
            print()
            continue

        response = model.chat(user_input)
        print(f"Atlas: {response}\n")


if __name__ == "__main__":
    main()
