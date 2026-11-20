# Versioning

PrimaGate uses semantic versioning. A version number is MAJOR.MINOR.PATCH:

- **MAJOR** goes up for a breaking change to the public surface below.
- **MINOR** goes up for a new feature that does not break the public surface.
- **PATCH** goes up for a fix that does not change the public surface.

**While the version is below 1.0, a minor version may break the public surface.** Pin a minor
version, for example `primagate>=0.1,<0.2`, until 1.0 is released. From 1.0 on, a breaking change to
the public surface needs a new major version.

## The public surface

These parts are covered by semantic versioning:

- **The calls and settings.** `completion` and `acompletion`, the `PrimaGate` class with its
  settings, and its methods. The module-level `primagate.drop_params` setting is covered.
- **The Policy config.** The `policy` setting, with its fields and their defaults (README step 15).
- **The Tier Binding config.** The `tier_bindings` setting, with its `model`, `settings`, `limits`
  and `pricing` fields (README step 14).
- **The response, chunk and metadata shapes.** The `ChatCompletion` a call returns, each
  `ChatCompletionChunk` a stream yields, and the `Metadata` on both: the fields they carry and what
  each means. The shapes are covered as the calls return them; these classes are not exported from
  `primagate`, so importing them from their module path is not covered.
- **The event types and their fields.** The event classes exported from `primagate`, each one's
  `event_type` name, and the fields it carries.
- **The exceptions.** The exceptions exported from `primagate`, and which one is raised for which
  failure.
- **The contracts you implement and pass in.** A `profile_store`, a `token_counter` and an
  `event_handler` are covered as the README documents them: their methods and how PrimaGate calls
  them. This holds even though their protocol classes are not importable from `primagate`.

Anything else exported from `primagate` itself (listed in `primagate.__all__`) is covered as well.
The `transport` setting is a test seam and is not covered.

## What counts as breaking

These changes are not breaking:

- Adding a field, an event type or an exception subclass. Adding an optional Policy field with a
  default that keeps today's behaviour is included; it does change the Policy `version` hash once.
  Your event handlers should ignore event types and fields they do not know.

These changes are breaking:

- Removing or renaming a field, an event type or an exception.
- Changing a field's type or meaning.
- Changing which exception a failure raises.
- Changing a documented default.
- Dropping a supported Python version.

## What is not covered

Anything under a module path not re-exported from `primagate` is internal, for example
`primagate.core` or `primagate.adapters`. It may change in any release, including a patch release.
Import what you use from `primagate` itself.

## Deprecation

The deprecation period is not yet set. Until it is, a feature may be removed in the next release
that is allowed to break it, without a deprecation warning first.
