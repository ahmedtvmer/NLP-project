# Project 2: Intelligent Search Engine

## 1. Objective
* Build a system that retrieves relevant documents based on a user query.
* Rank documents from most relevant to least relevant.
* Improve search quality using semantic understanding.

"Retrieving relevant documents" means:
Finding the most related texts to the user query based on meaning, not just exact matching words.

**Example of Search:**
Query: "food delivery problem"

Basic output (✗):
Returns documents that only contain the exact words "food", "delivery", "problem".

Improved output (✓) → you must do like this:
Returns documents such as:
* "The delivery was slow"
* "Late shipment issues"

Even if the exact words are not the same, the meaning is similar.

You must also implement:
* Ranking results (best results appear first)
* Top-k retrieval (e.g., top 5 results)
* Semantic similarity (understanding meaning, not just words)

## 2. Dataset: (From your choice)
Examples:
* News articles
* Wikipedia pages
* Product descriptions

## 3. Tasks
### ROLE 1. Preprocessing
* Convert text to lowercase
* Remove punctuation
* Remove stopwords
* Tokenization

### ROLE 2. Feature Extraction
* TF-IDF
* Word embeddings (Word2Vec / GloVe / BERT embeddings, ...)

### ROLE 3. Baseline Model
* Cosine similarity using TF-IDF.

### ROLE 4. Advanced Model
* Semantic search using embeddings
* OR Transformer-based embeddings (e.g., BERT)

## ROLE 5. Evaluation
* Evaluate search quality using Top-k results (e.g., Top-5)
* Use manual relevance checking:
  - For each query, check how many returned documents are actually relevant.
* Calculate **Precision@k**:
  - Precision@k = (Number of relevant documents in Top-k) / k

**Example:**
Query: "food delivery problem"
Top 5 results: ..... .
3 relevant → Precision@5 = 3/5

* Compare baseline vs advanced model:
  - Which model retrieves more relevant results?
  - Which model understands meaning better?
* Explain your results clearly

## ROLE 6. GUI

## ROLE 7. Report (write what you did in detail)
Include:
* Description of the system & Dataset used
* Preprocessing steps
* Models used
* Results and comparison
* Conclusion
