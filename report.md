# Design Decisions Report

## Embedding Model: BAAI/bge-small-en-v1.5

**Choice:** FastEmbed with `BAAI/bge-small-en-v1.5` (384 dimensions)

**Rationale:**
- **Lightweight**: Runs efficiently on CPU without GPU requirements
- **Quality**: Ranks among top models on MTEB benchmark for its size
- **Dimension**: 384-dim vectors balance storage efficiency with semantic richness
- **Local Processing**: No API calls needed, avoiding rate limits and costs

**Trade-offs:**
- Slightly lower accuracy than OpenAI `text-embedding-3-large` (1536 dims)
- CPU embedding adds ~250ms latency per query (acceptable for our use case)

## Vector Database: FAISS

**Choice:** FAISS with `IndexIVFFlat` (Inverted File Index)

**Rationale:**
- **In-Process**: No separate infrastructure to manage
- **Performance**: Sub-10ms median retrieval on 100K+ vectors
- **Simplicity**: Single file storage (`index.faiss`), easy to version and backup
- **Proven**: Battle-tested by Meta/Facebook at billion-scale

**Trade-offs:**
- Not distributed (single-node only)
- In-memory index requires sufficient RAM (~150MB for 100K vectors)
- No built-in persistence (we save/load manually)

**Why IVF over Flat?** With 100K+ vectors, brute-force search was ~40ms. IVF clustering reduces this to ~10ms with minimal accuracy loss (nprobe=10 searches 10% of clusters).

## Chunking Strategy

**Configuration:** 512 characters, 50-character overlap

**Rationale:**
- **512 chars**: Approximately 128 tokens — fits well within context windows while providing sufficient semantic content
- **Sentence-aware**: Splits on sentence boundaries first, then combines into chunks
- **50 char overlap**: Prevents information loss at chunk boundaries

**Trade-offs:**
- Smaller chunks (256) would improve retrieval precision but lose context
- Larger chunks (1024) would preserve context but reduce retrieval accuracy

## Telemetry Design

**Approach:** JSON-based metrics with merge-on-save

**Components:**
- `MetricsStore`: In-memory collection during runtime
- `metrics.json`: Persistent storage with automatic merging
- Streamlit dashboard: Real-time visualization

**Key Decisions:**
- **JSON over SQLite**: Simpler debugging, human-readable, sufficient for demo scale
- **Merge-on-save**: Each component appends to existing metrics (no overwrites)
- **Separate timing**: FAISS search timed separately from embedding for accurate reporting

## Agent Architecture

**Pattern:** Single-agent ReAct loop with function calling

**Why not multi-agent?**
- Simpler to implement and debug
- Sufficient for the travel planning use case
- Lower cost (fewer LLM calls)

The LLM acts as both reasoner and executor, calling tools as needed and synthesizing results into a coherent itinerary.

## Code Assistant: Self-Healing Loop

**Pattern:** Generate → Test → Parse Errors → Repair → Retry

**Key Decisions:**
- **Max 3 attempts**: Balances cost vs. success rate — most fixable errors resolve in 1-2 retries
- **Multi-language support**: Python (pytest), Rust (cargo test), JavaScript (jest)
- **Error extraction**: Language-specific regex parsing to feed precise errors back to LLM

**Why this approach?**
- **Iterative refinement**: LLMs often produce near-correct code; error feedback enables self-correction
- **Real execution**: Running actual tests catches runtime issues that static analysis misses
- **Streaming progress**: User sees each attempt in real-time

**Trade-offs:**
- Complex Rust code (borrow checker) may fail all 3 attempts — LLM limitation, not loop design
- Each retry costs additional tokens (~$0.01 per attempt)

## Future Improvements

1. **Persistent Vector DB**: Migrate to pgvector for production scalability and SQL integration

2. **Async Processing**: Implement async embedding and retrieval for better throughput

3. **Enhanced Evaluation**: Integrate RAGAS framework for comprehensive RAG metrics (faithfulness, answer relevancy)

4. **Multi-Agent Orchestration**: Use LangGraph for complex planning with specialized agents (flight agent, hotel agent, etc.)

5. **Caching**: Add Redis caching for frequent queries to reduce latency and cost

6. **Hybrid Search**: Combine vector similarity with BM25 keyword search for improved retrieval
