# Dataset Statistics - Member B

## Record Counts
- Train S1 / GT: 2,206,821
- Train S2: 5,034,616
- Train S3: 5,285,603
- Test S1: 1,732,544
- Test S2: 4,887,273
- Test S3: 5,082,316

## Singleton & Match Analysis
- Total S1 entities: 2,206,821
- Singletons (0 matches): 123,247
- Singleton rate: 5.58%

### Match distribution
| Number of matches | S1 entities |
|---:|---:|
| 0 | 123,247 |
| 1 | 119,157 |
| 2 | 375,212 |
| 3 | 530,841 |
| 4 | 484,115 |
| 5 | 321,957 |
| 6 | 164,868 |
| 7 | 63,968 |
| 8 | 18,680 |
| 9 | 4,205 |
| 10 | 534 |
| 11 | 37 |

- S1 entities with >=1 match: 2,083,574
- Average matches among matched S1 entities: 3.666
- Maximum matches: 11

## Country Distribution

### Train S1
| Country | Count |
|---|---:|
| US | 1,323,633 |
| India | 883,188 |

### Train S2 (300k sample)
| Country | Count |
|---|---:|
| US | 179,464 |
| India | 120,536 |

### Train S3 (300k sample)
| Country | Count |
|---|---:|
| US | 179,465 |
| India | 120,535 |

### Test S1
| Country | Count |
|---|---:|
| India | 809,986 |
| US | 663,106 |
| France | 259,452 |

- France appears in Test S1 and does not appear in the Train S1/S2/S3 country distributions.

## Name & Address Length Stats

### Train S1
| Statistic | Name length | Address length |
|---|---:|---:|
| Count | 2,206,821 | 2,206,821 |
| Mean | 24.034 | 52.066 |
| Std | 7.741 | 25.330 |
| Min | 3 | 11 |
| 25% | 18 | 33 |
| Median | 24 | 41 |
| 75% | 30 | 70 |
| Max | 105 | 256 |

### Train S2 (300k sample)
- Length statistics not required in the provided Phase 1 execution guide.

### Train S3 (300k sample)
- Length statistics not required in the provided Phase 1 execution guide.
