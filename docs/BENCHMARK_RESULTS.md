# SQL Agent

## First e2e test (first data ingestion)
* SQL agent model: qwen3.5 4B Q4_K_M (Dense, Thinking)
* LLM Evaluator model: qwen3.5 4B Q4_K_M (Dense, NO Thinking)
* Jaccard similarity score: not measured

### LLM evaluation
| Difficoltà | PASS | EMPTY | PARTIAL | ERROR |
| --- | --- | --- | --- | --- |
| **Easy** | 22 | 21 | 5 | 0 |
| **Medium** | 5 | 20 | 1 | 2 |
| **Hard** | 1 | 14 | 0 | 3 |
| **Impossible** | 1 | 4 | 0 | 1 |

**Notes:**
Local SLM has more issues with data structuring than anticipated, even if the documents are already fairly structured themselves.
Fully local ingestion is still possible, but requires breaking down documents intelligently into smaller chunks and then merge-map them.

## Data ingestion pipeline revision 1  (vision fallback for messy documents, structured import of all data)
* SQL agent model: gemma4 26B A4B QAT (MoE, Thinking)
* LLM Evaluator model: qwen3.5 4B Q4_K_M (Dense, NO Thinking)
* Jaccard similarity score: not measured

### LLM evaluation
| Difficoltà | PASS | EMPTY | PARTIAL | ERROR |
| --- | --- | --- | --- | --- |
| **Easy** | 36 | 5 | 6 | 1 |
| **Medium** | 16 | 7 | 5 | 0 |
| **Hard** | 0 | 18 | 0 | 0 |
| **Impossible** | 2 | 2 | 1 | 1 |


## Data ingestion pipeline revision 2 (improve data parsing guidelines, prompt engineering)
* SQL agent model: gemma4 26B A4B QAT (MoE, Thinking)
* LLM Evaluator model: qwen3.5 4B Q4_K_M (Dense, NO Thinking)
* Jaccard similarity score: 70.9651

### LLM evaluation
| Difficoltà | PASS | EMPTY | PARTIAL | ERROR |
| --- | --- | --- | --- | --- |
| **Easy** | 43 | 4 | 1 | 0 |
| **Medium** | 21 | 1 | 6 | 0 |
| **Hard** | 2 | 16 | 0 | 0 |
| **Impossible** | 3 | 1 | 2 | 0 |

## Database schema revision (simplify structure for techniques and licenses, add DB metadata, prompt engineering)
* SQL agent model: gemma4 26B A4B QAT (MoE, Thinking)
* LLM Evaluator model: qwen3.5 4B Q4_K_M (Dense, NO Thinking)
* Jaccard similarity score: 79.0540

### LLM evaluation
| Difficoltà | PASS | EMPTY | PARTIAL | ERROR |
| --- | --- | --- | --- | --- |
| **Easy** | 45 | 3 | 0 | 0 |
| **Medium** | 20 | 1 | 7 | 0 |
| **Hard** | 9 | 6 | 3 | 0 |
| **Impossible** | 3 | 1 | 2 | 0 |

# SLM testing 
The real question at this stage is: how does a small-sized model suitable for on-device performance currently perform in benchmarks?

## Phone tier
* SQL agent model: gemma4 E2B Q4_K_M (PLE, Thinking)
* LLM Evaluator model: qwen3.5 4B (Dense, NO Thinking)
* Jaccard similarity score: 22.2484

### LLM evaluation
| Difficoltà | PASS | EMPTY | PARTIAL | ERROR |
| --- |------|-------|---------|-------|
| **Easy** | 19   | 25    | 2       | 2     |
| **Medium** | 3    | 23    | 0       | 2     |
| **Hard** | 15   | 1     | 1       | 1     |
| **Impossible** | 1    | 1     | 1       | 3     |

**Notes:**
Several issues following SQL formatting instructions, not viable at this stage.
SQL agent should be simplified and broken into multiple steps.

## Average consumer GPU tier
* SQL agent model: qwen3.5 4B Q4_K_M (Dense, NO Thinking)
* LLM Evaluator model: qwen3.5 4B (Dense, NO Thinking)
* Jaccard similarity score: 39.0859

### LLM evaluation
| Difficoltà | PASS | EMPTY | PARTIAL | ERROR |
| --- |------|-------|---------|-------|
| **Easy** | 31   | 15    | 1       | 1     |
| **Medium** | 5    | 19    | 1       | 3     |
| **Hard** | 3    | 12    | 0       | 3     |
| **Impossible** | 1    | 4     | 0       | 1     |

**Notes:**
Issues stem primarily from DDL understanding, viable but more metadata and prompting is needed.
Will 100% benefit from SQL agent simplification.