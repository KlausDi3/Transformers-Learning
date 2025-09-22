# Study Notes: Tokenizer

## 1. Basic Concepts

- **Tokenizer function**: Convert natural language into numeric format for the model.
- **Token**: Not always equal to a word; can be a character or a subword depending on the algorithm.
- **Common algorithms**: WordPiece, BPE, SentencePiece.
- **Why it matters**: Training and inference both require consistent encoding.

---

## 2. Loading a Tokenizer

**Example**: `AutoTokenizer.from_pretrained("bert-base-uncased")`

- `use_fast=True` → use the Rust-implemented fast tokenizer.

### Difference:
- **Fast version**: is faster, provides offset mapping.
- **Slow version**: is Python-based, full-featured but slower.

---

## 3. Basic Operations

### 3.1 Tokenize
```python
tokenizer.tokenize("I love NLP")
```
**Output**: `["i", "love", "nl", "##p"]`

### 3.2 Encode
```python
tokenizer.encode("I love NLP")
```
**Output**: `[101, 1045, 2293, 17953, 2361, 102]`  
CLS = 101, SEP = 102.

### 3.3 Batch Encode
```python
tokenizer(["I love NLP", "Hello world"], padding=True, truncation=True)
```
**Output includes**: `input_ids`, `attention_mask`, `token_type_ids`.

### 3.4 Decode
```python
tokenizer.decode([101, 1045, 2293, 102])
```
**Output**: `"[CLS] i love [SEP]"`  
With `skip_special_tokens=True` → `"i love"`

---

## 4. Output Structure

### 4.1 input_ids
- Numeric IDs corresponding to tokens.
- Used as the model's main input.
- **Example**:
  - Tokens: `["i", "love", "nl", "##p"]`
  - input_ids: `[1045, 2293, 17953, 2361]`

### 4.2 attention_mask
- Sequence of 0s and 1s, same length as input_ids.
- 1 = valid token, 0 = padding.
- **Example**:
  - input_ids: `[1045, 2293, 17953, 2361, 0, 0]`
  - attention_mask: `[1, 1, 1, 1, 0, 0]`

### 4.3 token_type_ids (segment IDs)
- Marks which sentence each token belongs to.
- Used in sentence pair tasks (e.g., NLI, QA).
- 0 = first sentence, 1 = second sentence.
- **Example**:
  - Input: `"I love NLP [SEP] It is powerful"`
  - token_type_ids: `[0, 0, 0, 0, 0, 1, 1, 1]`

---

## 5. Differences Between Methods

### tokenize vs encode
- **tokenize** → token strings
- **encode** → token IDs

### encode vs tokenizer()
- **encode** → returns a list
- **tokenizer()** → returns BatchEncoding (with IDs, mask, types), more complete

### Single vs batch
- **Single** → good for debugging
- **Batch** → efficient for training, supports padding/truncation

---

## 6. Key Notes

- **Max length**: Models have a limit (e.g., BERT 512). Longer sequences must be truncated.
- **Padding**: Usually added on the right; must match pretraining setup.
- **Special tokens**: Avoid adding them twice.
- **New tokens**: Expanding the vocabulary requires resizing the embedding layer.
- **Decoding**: Use `skip_special_tokens=True` for clean text output.

---

## 7. Example

```python
text = "Transformers are powerful"
tokens = tokenizer.tokenize(text)
ids = tokenizer.encode(text)
batch = tokenizer([text], padding=True, truncation=True, return_tensors="pt")
```

### Results:
- **tokens**: `["transform", "##ers", "are", "powerful"]`
- **ids**: `[101, 10938, 2015, 2024, 3928, 102]`
- **batch**:
  - input_ids: `[[101, 10938, 2015, 2024, 3928, 102]]`
  - attention_mask: `[[1, 1, 1, 1, 1, 1]]`
  - token_type_ids: `[[0, 0, 0, 0, 0, 0]]`
