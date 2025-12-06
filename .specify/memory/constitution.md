<!-- SYNC IMPACT REPORT
Version change: N/A (initial version) → 1.0.0
List of modified principles: N/A
Added sections: All principles and sections based on user input
Removed sections: N/A
Templates requiring updates:
- .specify/templates/plan-template.md ✅ updated / ⚠ pending
- .specify/templates/spec-template.md ✅ updated / ⚠ pending
- .specify/templates/tasks-template.md ✅ updated / ⚠ pending
- .specify/templates/commands/*.md ✅ updated / ⚠ pending
Follow-up TODOs: None
-->
# AI / Spec-Driven Book Creation using Spec-Kit Plus, Claude Code, and Docusaurus Constitution

## Core Principles

### Spec-First Development
All content must be derived from Constitution → Spec → Plan → Tasks; Determinism over improvisation: AI output must follow defined structure, not ad-hoc generation

### Clarity and Pedagogy
Content must be clear, progressive, and instructional; Clear technical writing suitable for readers with basic software engineering knowledge; No filler, marketing language, or vague explanations; Each section must have a defined intent and learning outcome; Terminology must be consistent across the entire book

### Tool Transparency and Reproducibility
Every tool (Spec-Kit Plus, Claude Code, Docusaurus) must be explained in context; Reproducibility: Another team should be able to recreate the book using the same process; Content generation must be compatible with Claude Code via Cloud CLI

### AI Behavior Rules
AI must not skip lifecycle stages; AI must not introduce features or ideas outside the approved specification; AI must explicitly respect constraints defined in each phase; AI must prefer correctness and structure over verbosity; AI must not hallucinate tools, APIs, or unsupported features

### Platform & Tooling Constraints
Documentation framework: Docusaurus (mandatory); Output format: Markdown / MDX compatible with Docusaurus; Repository structure must follow Docusaurus best practices; Deployment target: GitHub Pages

### Quality and Validation
Each section must map back to the specification; Navigation must work correctly in Docusaurus; Book builds successfully with no broken links; Content is coherent when read top-to-bottom; The spec-driven workflow is clearly observable in the final output

## Writing & Content Standards

Clear technical writing suitable for readers with basic software engineering knowledge; No filler, marketing language, or vague explanations; Each section must have a defined intent and learning outcome; Terminology must be consistent across the entire book; All examples must align with the Spec-Kit Plus workflow

## Development Workflow and Quality Gates

Spec-First Development: All content must be derived from Constitution → Spec → Plan → Tasks; Determinism over improvisation: AI output must follow defined structure, not ad-hoc generation; Quality Gates: Each section must map back to the specification, Navigation must work correctly in Docusaurus, Book builds successfully with no broken links, Content is coherent when read top-to-bottom, The spec-driven workflow is clearly observable in the final output

## Governance

All outputs strictly follow the user intent; Prompt History Records (PHRs) are created automatically and accurately for every user prompt; Architectural Decision Record (ADR) suggestions are made intelligently for significant decisions; All changes are small, testable, and reference code precisely; All content must map back to specification; Book must build successfully with no broken links; The spec-driven workflow must be clearly observable in the final output

**Version**: 1.0.0 | **Ratified**: 2025-12-06 | **Last Amended**: 2025-12-06