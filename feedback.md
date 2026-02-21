# 1.5 Composer Feedback

* **Model is frequently extremely overconfident**
    * Hallucinated version numbers for `requirements.txt`
    * Called non-existent functions
    * Just flat out didn't perform requested tasks even when given detailed & thorough instructions & feedback
* **Model assumes alignment, even through ambiguity**
    * Does not clarify user requests when they are vague, just charges ahead which leads to increasing user-model divergence over time.

---

### Cost

| Model | Input/Mtok | Out/Mtok | Context |
| :--- | :--- | :--- | :--- |
| Com. 1.5 | $3.5 | $17.5 | |
| Opus 4.6 | $5 | $25 | |
| Sonnet 4.6 | $3 | $15 | |
| 3.1 Gen-Pro | $2 | $12 | < 200k Tok |
| 3.1 Gen-Pro | $4 | $18 | > 200k Tok |

---

### Cost/Intelligence Ratio

* At a cost that is comparable to Gemini 3.1 Pro, and greater than Sonnet 4.6 Pro, I struggle to recommend 1.5 Composer. Put simply, I do not trust it to accomplish tasks. While fast, its numerous drawbacks make the user experience a frustrating one.
* To be quite frank, the model does not have the intelligence of other frontier models, but is priced as one, making for a very difficult value proposition for the end user.