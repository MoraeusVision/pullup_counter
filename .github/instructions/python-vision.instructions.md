---
description: "Use when writing or refactoring Python computer-vision code in this repo. Enforces scalable-but-simple design, static typing, supervision-first annotation workflows, and no ultralytics usage."
name: "Python Vision Standards"
applyTo: "**/*.py"
---
# Python Vision Standards

- Prefer scalable design that stays simple:
  - Use classes when state or extension points are needed.
  - Keep small, focused functions for isolated logic.
  - Avoid over-engineering (no unnecessary abstraction layers).
- Use typing consistently:
  - Add type hints for function params and return values.
  - Use typed containers (`list[str]`, `dict[str, float]`) and explicit `Optional`/union where relevant.
  - Prefer `TypedDict` or `dataclass` for structured data passed between components.
- Use `supervision` for frame annotation and visualization tasks:
  - Prefer `supervision` annotators/utilities over ad hoc OpenCV drawing when feasible.
  - Keep annotation code separate from model inference and counting/business logic.
- Do not use `ultralytics`:
  - Do not add `ultralytics` as a dependency.
  - Do not introduce imports from `ultralytics`.
  - If an example includes `ultralytics`, adapt it to the existing stack (`rfdetr`, `inference`, `supervision`).
