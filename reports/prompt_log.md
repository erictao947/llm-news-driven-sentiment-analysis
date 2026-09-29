# Prompt design log (in-sample only, frozen before any OOS scoring)

| Version | Change | Spot check (50 random in-sample events, seed 20250901) |
|---|---|---|
| v1 | Rubric, category definitions, 30-name universe reference, 12 hand-written examples; 4,980 tokens | ~45/50 labels judged right. Repeated misses: analyst roundups ("10 Analysts Assess Nike") and service outages (AWS Downdetector) labeled `other`. |
| v2 | Two mapping rules: roundups -> `analyst_rating`; outages and product bugs -> `product` | Exactly those 2 labels changed; the other 48 scores and categories were unchanged. **Frozen.** |

Spot-check tables: `llm_spotcheck.md` (v2). No return data was looked at while designing the prompt.
