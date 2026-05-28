# Optimized Plan

- Run rocm-sanitizer modes over minimal shapes that expose races, sync bugs, uninitialized reads, and memory errors.
- Attach only evidence that was actually collected.
- Keep compile-only, timing-only, blocked-profile, and counter-backed labels distinct.
