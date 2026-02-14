# Role Taxonomy Ensemble Engine

Hierarchical LLM-powered role taxonomy classifier for multi-field role harmonization.

## Features
- Multi-field classification (role_title, job_title, vendor_role)
- Top-3 candidate extraction per field
- Weighted ensemble aggregation
- Family-first hierarchical scoring
- Confidence + margin-based review logic
- CSV input/output pipeline

## Architecture

See system design diagram below.

```mermaid
flowchart TD
  A[Input records<br/>username, role_title, job_title, vendor_role] --> B[Per-field classifier (LLM)]
  B --> C1[role_title -> top-k canonical roles + confidence]
  B --> C2[job_title -> top-k canonical roles + confidence]
  B --> C3[vendor_role -> top-k canonical roles + confidence]

  C1 --> D[Family-first aggregation]
  C2 --> D
  C3 --> D

  D --> E[Decision logic]
  E --> F[Final family]
  F --> G[Select canonical role within family]
  G --> H[Output CSV + diagnostics]