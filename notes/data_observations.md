# Member C — Data Observations

## Positive Pair Inspection

### Scope

Manually inspected 60 positive entity groups from the training ground truth.
The purpose was to understand how the same real-world business/entity can be represented
differently across Source 1, Source 2, and Source 3.

The inspection shows that true matches can contain substantial variation in both
business names and addresses.

---

## 1. Business Name Noise

Observed examples include:

- `Advanced Circle Group` → `Advanced Cilce Group`
- `Bay Disciplined` → `Bay Discip1ined`
- `Grand Farms Private Limited` → `Grand Farms Pirnavte Limited`
- `Osprey & Sons Limited` → `Osprey & S0ns`

Observed patterns:

- spelling errors
- character substitutions
- OCR-like errors
- case differences
- punctuation differences
- visually similar character substitutions such as `l → 1` and `o → 0`

These variations mean exact business-name matching is insufficient.

---

## 2. Extra, Missing, or Modified Name Tokens

Examples:

- `J 8 Innovative Motors LLC`
  → `J 8 MOTORS INNOVATIVE INNOVATIVE LLC`

- `Lina Garvin Martin Inc`
  → `Lina Garvin Inc Partners`

- `Vishranti Marketing Mumbai`
  → `Vishranti Mumbai Marketing`

- `Osprey & Sons Limited`
  → `Osprey Sons Limited Services`

Observed patterns:

- token reordering
- duplicated tokens
- removed tokens
- additional tokens
- substituted tokens

Token-level similarity is therefore important in addition to character similarity.

---

## 3. Legal Suffix Variation

Examples:

- `Advanced Circle Group`
  → `Advanced Circle LLC`

- `United Bny Clinic`
  → `United Bny Clinic Co`

- `United Bny Clinic`
  → `United Bny Clinic Ltd`

- `Bright Seafood Inc.`
  → `Bright Seafood Incorporated`

- `Yellow Deals Private Limited`
  → `Yellow Deals Private Limited`

- `Anand United Global Private Limited`
  → `ANAND UNITED GLOBAL PRIVATE LTD`

Observed variations include:

- LLC
- Inc / Incorporated
- Ltd
- LP
- Co
- Pvt Ltd
- Private Limited

Legal suffixes should not dominate entity similarity.

---

## 4. Partial and Abbreviated Business Names

Examples:

- `Primary Care Clinic LLC`
  → `Primary Care`

- `Celestial Memorial Trust`
  → `Celestial Memorial`

- `Jemmott Tax Service Corp`
  → `Jemmott Tax`

- `Osprey & Sons Limited`
  → `Osprey`

These are positive matches despite substantial name truncation.

---

## 5. Domain and Website-Style Names

Examples:

- `Lina Garvin Martin Inc`
  → `linagarvinmartin.com`

- `Teamsters Local 425`
  → `teamsterslocal425.com`

- `Jackson, Hall and Bernardo`
  → `JACKSONHALLBERNARDO.COM`

- `Behavioral Health Clinic`
  → `BEHAVIORALHEALTHCLINIC.COM`

- `Vadodara Industries Pvt Ltd`
  → `vadodaraindustries.com`

- `Viraaj Brokerage (India) Corporation`
  → `VIRAAJBROKERAGEINDIA.COM`

- `Celestial Memorial Trust`
  → `celestialmemorialtrust.com`

Domain-style representations should therefore be normalized before comparison.

---

## 6. Alternate / DBA-Style Names

Some positive matches have names that are substantially different from the
Source 1 representation.

Examples include:

- `Hutcherson Investments`
  → `Novinoviaria`

- `Kozlowski, Lopez & Smith LLC`
  → `Ciralumyuma`

These cases demonstrate that name similarity alone can fail badly.

Address and contextual signals become especially important for such matches.

---

## 7. Multilingual and Multi-Script Representations

Example:

`Anand United Global Private Limited`

→

`ಆನಂದ್ ಯುನೈಟೆಡ್ ಗ್ಲೋಬಲ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್`

The underlying entity is the same even though the script differs.

Other observations included multilingual address representations.

This indicates that multilingual and cross-script normalization is important for
entity resolution.

---

## 8. Accents and Diacritics

Examples:

- `Vishranti Marketing Mumbai`
  → `Vishranti Márketing Mumbai`

- `Bay Disciplined`
  → `BAY DÍSCIPLINED`

- `Jemmott Tax Service Corp`
  → `Jemmott Táx Service Corp`

- `Jemmott Tax Service Corp`
  → `Jemmott Tax Sérvice Corp`

Diacritics should generally be normalized so that accented and unaccented
representations can be compared.

---

## 9. Address Abbreviations and Formatting

Examples included:

- `Road` → `Rd`
- `Lane` → `Ln`
- `Trail` → `Trl`
- punctuation changes
- whitespace changes
- `#` and unit formatting differences
- PO Box / PMB representations

Example:

`Jemmott Tax Service Corp`
→ `Jemmott Tax Sérvice Corp` with `Lane → Ln`.

Address normalization should therefore standardize common abbreviations and
formatting.

---

## 10. Address Component Reordering

The same address can appear with components in different orders.

Examples were observed for:

- street
- city/locality
- state
- postal information
- unit/PMB information

This means raw address-string equality is unreliable.

---

## 11. Address Truncation and Missing Components

Examples:

- `Teamsters Local 425` with a complete address
  → another representation with missing address information.

- `Bright Seafood Inc.`
  → `Bright Seafood` with some address components missing.

- `Osprey & Sons Limited`
  → `Osprey` with a full address.

Missing fields should therefore be treated as missing evidence rather than
automatically as evidence of a mismatch.

---

## 12. Address Character and OCR-Like Noise

Examples included:

- spelling corruption in locality/street names
- character substitutions
- formatting corruption

Example:

`Jemmott Tax Service Corp`
→ address containing `Prospect` represented as `Propect`.

Another example included locality spelling variation such as:

`New Richmond` with a corrupted representation.

---

## 13. Address Number Variations

Observed examples include:

- `351` → `35`
- `715` → `0715`

These show that numeric components can also contain noise.

Therefore, address-number comparison should allow controlled variation rather than
requiring exact string equality.

---

## 14. Address Ranges

Example:

`Barney & Stockton Lending`

had representations containing:

- `#19705`
- `19705-19707`

Address ranges can therefore represent the same entity/location even when the
numeric strings differ.

---

## 15. Different Locality Names

A positive match can contain different locality descriptions.

Example:

`New Kent County`
vs.
`Providence Forge`

Another observed example involved:

`Islip`
vs.
`Brentwood`

Locality differences should therefore not automatically eliminate a candidate.

---

## 16. Structured Indian Addresses

Several positive examples contained Indian address structures involving:

- state abbreviations
- city/locality
- district
- postal information
- multilingual state names

Example:

`Yellow Deals Private Limited`

had an address representation involving:

`Disa / Banas Kantha / Gujarat`

and another representation containing the Gujarati form of Gujarat.

This suggests that address parsing should consider structured components rather than
only whole-string similarity.

---

## 17. Generic Business Names

Examples such as:

- `South College`
- `South Services`
- `Osprey`

show that short or generic names can have weak discriminative power.

For such entities, address and other contextual features become more important.

---

## 18. Strong Address Signal Despite Weak Name Similarity

Important positive examples include:

`South College`
→ `South Services`

and:

`Hutcherson Investments`
→ `Novinoviaria`

In these cases, the business names are weakly similar or substantially different,
while the address provides strong evidence.

This demonstrates why the system should combine name and address signals.

---

## 19. Professional Titles and Credentials

Example:

`Gwen R. Stagner, O.D.`

appeared as:

`Gwen R. Stagner, O.D. Inc.`

and another representation had punctuation/reordering differences.

Professional credentials and titles can therefore appear as part of the
business/entity name and should not automatically be treated as mismatches.

---

## 20. Noisy Additional Information

Some records contain additional information such as:

- website information
- phone numbers
- unit numbers
- PO Box information
- professional credentials

These fields can be useful supporting signals but may also introduce noise.

---

## 21. Multiple Representations for One Entity

A single Source 1 entity can map to multiple Source 2/Source 3 representations.

For example:

`Bright Seafood Inc.`

had several positive representations involving:

- `BRIGHT-SEAFOOD`
- `Bright Seafood`
- `Bright Seafhiigod (Inc.)`
- `Bright Seafood Incorporated`
- `Bright Inc. Seafood`

Similarly, several representations were observed for:

- `Teamsters Local 425`
- `Vishranti Marketing Mumbai`
- `Anand United Global Private Limited`
- `Osprey & Sons Limited`

This means the model must handle one-to-many representations rather than assuming
one canonical string per entity.

---

## 22. Country as Supporting Context

Country information can provide useful contextual evidence.

However, the positive-pair inspection suggests that country should be treated as
a supporting signal rather than as the only matching criterion.

---

# Overall Conclusions

The 60 manually inspected positive groups show that genuine entity matches can
have substantial differences in both business name and address.

The major sources of positive-pair variation were:

1. spelling and character corruption
2. OCR-like substitutions
3. legal suffix variation
4. token addition, deletion, duplication, and reordering
5. partial/truncated names
6. domain-style business names
7. alternate/DBA-style names
8. multilingual and multi-script representations
9. accents and diacritics
10. address abbreviation and formatting differences
11. address component reordering
12. address truncation and missing fields
13. address character noise
14. address-number variation
15. address ranges
16. locality/city variation
17. structured Indian addresses
18. generic business names
19. strong-address / weak-name matches
20. professional titles and additional information
21. multiple representations for the same entity

## Key Takeaway

No single field is consistently reliable.

A robust entity-resolution system should combine complementary signals from
business names, addresses, country/context, and learned representations.

In particular, the inspection provides evidence for using both classical
normalization/string features and embedding-based representations. Embeddings
should be evaluated carefully on the difficult cases identified above, especially
multilingual representations, domain names, truncation, noisy strings, and
strong-address/weak-name matches.

