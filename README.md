# Transformers-Learning

## 1. Environment Setup

### 1.1 Required Libraries
- **transformers**: HuggingFace core library  
- **PyTorch** or **TensorFlow**: backend deep learning framework  
  - Recommended: **PyTorch** (most official examples use it)

---

### 1.2 Installation Commands

#### Install with PyTorch (recommended)
```bash
pip install transformers torch
```

#### Install with TensorFlow
```bash
pip install transformers tensorflow
```

---

### 1.3 GPU Support
**If you have an NVIDIA GPU:**
- Install CUDA drivers + the correct PyTorch CUDA version
- Example (CUDA 12.1):
  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cu121
  ```

**If you don't have a GPU:**
- You can still run on CPU, but it will be slower

---

### 1.4 Version Compatibility
Always check that your transformers version matches the installed backend (torch / tensorflow).

**Example**: `transformers==4.44.0` works with `torch>=2.2.0`.

**To check versions:**
```bash
pip show transformers
pip show torch
```

---

### 1.5 Optional Tools

#### Datasets (for loading benchmark datasets):
```bash
pip install datasets
```

#### Accelerate (for hardware acceleration, distributed training):
```bash
pip install accelerate
```

#### Sentencepiece / Tokenizers (for multilingual models):
```bash
pip install sentencepiece
pip install tokenizers
```

---

### 1.6 Common Issues
- ❌ **Error: CUDA not available** → Check driver and CUDA toolkit installation
- ❌ **Version mismatch** → Align torch and transformers versions  
- ❌ **Tokenizer errors** → Ensure proper installation of tokenizers or sentencepiece

---

## 2. What is a Pipeline
- **Definition**: A Pipeline is a high-level interface provided by HuggingFace to quickly use pretrained models.  
- **Purpose**: It bundles the entire workflow so you can go from input to output with just a few lines of code.  
- **Core idea**: Experience NLP / CV / Speech models easily without dealing with low-level implementation details.  

---

### 2.1 Supported Tasks (Detailed)

Pipelines support many **common NLP tasks**, as well as vision, speech, and multimodal tasks.  

#### 2.1.1 Text Tasks
1. **Sentiment Analysis (sentiment-analysis)**  
   - Function: Detect if text sentiment is positive or negative  
   - Use cases: Product reviews, customer feedback, opinion mining  
   - Output: Label + confidence score  

2. **Text Classification (text-classification)**  
   - Function: Assign one or multiple predefined categories to a text  
   - Use cases: News categorization, spam detection  
   - Output: Category labels with probabilities  

3. **Question Answering (question-answering)**  
   - Function: Answer a question given a context passage  
   - Input example:  
     ```python
     {"question": "Who is the president of the US?", "context": "Joe Biden is the president of the United States."}
     ```  
   - Use cases: Knowledge-base Q&A, search augmentation  
   - Output: Extracted answer span, start/end positions, confidence  

4. **Text Generation (text-generation)**  
   - Function: Generate text continuation from a prompt  
   - Use cases: Story generation, dialogue systems, writing assistants  
   - Parameters: Control max length, sampling strategy, etc.  

5. **Fill-Mask (fill-mask)**  
   - Function: Predict the masked word `[MASK]`  
   - Use cases: Cloze tasks, language modeling, word suggestion  
   - Output: Candidate words with probabilities  

6. **Summarization (summarization)**  
   - Function: Generate a summary from a long text  
   - Use cases: News highlights, document compression  
   - Output: Concise natural language summary  

7. **Translation (translation)**  
   - Function: Translate text from one language to another  
   - Use cases: Cross-language communication, content localization  
   - Output: Translated text  

#### 2.1.2 Multimodal Tasks
- **Image Classification (image-classification)**  
- **Object Detection (object-detection)**  
- **Automatic Speech Recognition (ASR)**  
- **Multimodal inputs (e.g., text + image)**  

> 📌 **Beginner Tip**: Start with simple tasks like sentiment analysis, classification, or fill-mask. Move on to generation, summarization, and translation later.

---

### 2.2 Usage Example

#### Install dependencies
```bash
pip install transformers torch
```

#### Import the pipeline function
```python
from transformers import pipeline
```

#### Specify a task
```python
classifier = pipeline("sentiment-analysis")
```

#### Input data
```python
result = classifier("I love HuggingFace Transformers!")
```

#### Output result
```python
[{'label': 'POSITIVE', 'score': 0.99}]
```

---

### 2.3 Common Parameters
- **model**: specify the pretrained model to use
- **device**: choose hardware (CPU: -1, GPU: 0,1,...)
- **max_length**: maximum length for text generation
- **batch_size**: batch size for efficiency
- **truncation / padding**: handle long inputs

---

### 2.4 Internal Workflow (Detailed)
A Pipeline follows a standardized data flow:

#### 2.4.1 Preprocess
- **Function**: Convert user input into a format the model understands
- **Examples**:
  - Text → tokenizer → token IDs
  - Image → resize, normalization
  - Audio → feature extraction (MFCC, spectrogram)

#### 2.4.2 Model Inference (Forward)
- **Function**: Pass preprocessed input into the model to get raw outputs (logits / embeddings)
- **Process**:
  - Transformer layers perform computations
  - Outputs hidden states, attention weights, prediction scores

#### 2.4.3 Postprocess
- **Function**: Convert raw model outputs into human-readable results
- **Examples**:
  - Classification: pick the label with highest probability
  - Text generation: decode tokens into natural text
  - Question answering: extract answer span from context

#### 2.4.4 Final Output
- Returns results as a Python dictionary or string
- **Example**:
  ```python
  [{'label': 'POSITIVE', 'score': 0.998}]
  ```

⚡ **Think of a Pipeline as a black box**: input natural data, output final answers.  
Internal steps are **Preprocess → Model → Postprocess**.

---

### 2.5 Advantages
- Quick to start, requires minimal code
- Supports many tasks and pretrained models
- Unified interface lowers the learning barrier

---

### 2.6 Limitations
- Default models may not fit all tasks or languages
- Large models can be slow and memory intensive
- Some tasks (e.g., QA) require structured input formats
- Deep customization requires advanced knowledge

---

### 2.7 Beginner Learning Path
1. Start with sentiment analysis and text classification
2. Learn to switch models (e.g., BERT, GPT2)
3. Try more advanced tasks (generation, summarization, translation)
4. Practice parameter tuning (max_length, device, batch_size)
5. Explore fine-tuning and custom Pipelines for deeper use
