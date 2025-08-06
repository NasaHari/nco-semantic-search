# Evaluation Report

## Precision@K
- Test Setup: 50 sample queries from NCO examples.
- Precision@5: [Calculate and insert, e.g., 0.85]
- Notes: Manual validation against ground truth matches.

## Latency Benchmarks
- Average Query Time: [Benchmark, e.g., 0.2s on CPU]
- Hardware: [Specify, e.g., Intel i7, 16GB RAM]

## Test Cases
- Query: "doctor" → Expected: Medical codes with high scores.
- Edge Case: Misspelled "docter" → Should still match semantically.
- No Match: "alien spaceship pilot" → Low scores or fallback message.
