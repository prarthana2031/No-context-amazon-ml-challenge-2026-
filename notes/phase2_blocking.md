# Phase 2 - Classical Multi-Key Blocking

## Objective

Design and evaluate classical blocking keys for reducing the S1-to-S2/S3 candidate search space while preserving as many known true matches as possible.

## Dataset and Evaluation

- Positive S1-to-S2/S3 pairs: 7,638,365
- Country is used as a hard filter.
- Recall is calculated as:

  `blocking hits / total positive pairs`

A hit means that a known ground-truth S1-to-S2/S3 pair satisfies the country filter and the corresponding blocking-key condition.

## Blocking Keys

### 1. Name Prefix
First 5 characters of the normalized business name.

### 2. Exact Name
Business name normalized by lowercasing and removing non-alphanumeric characters.

### 3. Sorted-Token Name
Business-name tokens are normalized and sorted before comparison. This helps when the same name contains tokens in a different order.

### 4. Phonetic Name
A lightweight Soundex-style encoding is generated from alphabetic name tokens to capture some spelling/phonetic variation.

### 5. Street Number
The leading numeric component of the business address.

### 6. Postal Code
A 5-digit US ZIP code or 6-digit Indian PIN code extracted from the address.

## Recall Results

| Blocking key | Hits | Recall |
|---|---:|---:|
| Name prefix | 5,681,961 | 74.3871% |
| Exact name | 1,752,581 | 22.9445% |
| Sorted-token name | 2,055,587 | 26.9113% |
| Phonetic name | 1,841,392 | 24.1071% |
| Street number | 2,714,734 | 35.5408% |
| Postal code | 366,503 | 4.7982% |
| **Combined classical blocking** | **6,401,011** | **83.8008%** |

## Combined Blocking

The combined classical blocker takes the union of the individual blocking conditions after applying the country hard filter.

It captures:

- 6,401,011 of 7,638,365 positive pairs
- Combined recall: 83.8008%

Therefore, the classical blocking stage alone does not capture every known positive pair. The remaining positives are candidates for additional blocking strategies, including embedding/ANN blocking handled by the corresponding team member.

## Phase 2 Conclusion

The classical multi-key blocking implementation provides multiple complementary signals rather than relying on a single blocking key. Name prefix provides substantial recall, while street number and the name-based keys contribute additional coverage. Postal-code blocking has lower standalone recall but remains useful as a complementary high-specificity key.

All measured results are saved in:

`notes/phase2_blocking_recall.csv`
