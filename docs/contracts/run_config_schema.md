# Contract: run config schema

- version: v1
- owner: mantis.config.schema
- status: LIVE since scaffold (WP0)

## Summary
pydantic models, extra=forbid; schema_version key in every file

## Who asserts what where
| assertion | where |
|---|---|
| missing key = hard error; unknown key = hard error (top-level AND nested) | mantis.config.loader / mantis.config.schema (StrictModel: extra="forbid", frozen) |
| schema_version pinned to the exact current version (SCHEMA_VERSION = 1) | mantis.config.schema field validator |
| identity keys (encoding, representation) have no terminal defaults; representation is a closed Literal | mantis.config.schema IdentityConfig |
| configs are complete and minted, never hand-varied; delta stamped in header | tools/mint_config.py |
| two configs differ exactly where claimed | tools/config_diff.py |
| every committed config schema-validates (empty set = gate failure) | CI gate 7 (tools/ci_gates/validate_configs.py) |

## Pinning tests
| test | file |
|---|---|
| example config validates | tests/config/test_schema.py |
| top-level / nested unknown key rejected | tests/config/test_schema.py |
| missing top-level / identity key rejected | tests/config/test_schema.py |
| wrong schema_version rejected | tests/config/test_schema.py |
| representation "grid" (or anything outside the closed set) rejected | tests/config/test_schema.py |
| all model fields required (no code-side defaults; model_fields introspection) | tests/config/test_schema.py |
| mint output validates; header stamped; unknown delta key exits 2 | tests/config/test_mint_and_diff.py |
| diff exit 0 on exactly-claimed diff; exit 1 on unclaimed / identical-claimed | tests/config/test_mint_and_diff.py |
