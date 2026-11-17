# File-backed Profile store

By default an instance keeps its Profile in memory, so a restart starts cold. `FileProfileStore`
keeps it in one file, so a Profile whose fingerprint still matches is reused after a restart:

```python
import primagate

gate = primagate.PrimaGate(profile_store=primagate.FileProfileStore("profile.json"))
```

Only background preparation reads or writes the file (README step 18): it reads it once when it
starts, and writes it when a new Profile is ready.

The path may be text or a path object. An empty path, a bytes path, a path object that gives bytes,
or anything else is refused with `BadRequestError` when the store is built. A relative path is
fixed against the working directory at that moment, so a later change of directory does not move
the file. A missing parent directory is created on the first write.

Point the store at the real file, not a symbolic link: a write replaces a link at the path with a
regular file.

## Format

The file is UTF-8 JSON holding exactly `Profile.to_dict()`: the Profile's six fields with keys
sorted, indented by two spaces and ending in a newline. Text outside ASCII is written as `\u`
escapes. There is no envelope around it and only one Profile per file.

## Versioning

The file is versioned by the Profile's own `schema_version`. There is no second version number and
no migration step:

- When the file's `schema_version` or input fingerprint differs from the current one, PrimaGate
  treats it as a miss and does not use it.
- A file that cannot be read as a Profile is also a miss: invalid JSON, invalid UTF-8, a truncated
  file, an unknown or missing field.
- After a miss, the next Profile written replaces the file. That includes a file edited by hand and
  saved with a byte order mark or as invalid JSON.

An OS error other than a missing file is not a miss. If the path is a directory, or the file cannot be read for lack of
permission, the error reaches the caller unchanged, so a wrong path is noticed instead of looking
like a cold start.

## Atomic replace

A write goes to a temporary file in the same directory. It is flushed, passed to `fsync`, then moved
over the target with `os.replace`. A reader sees the old Profile or the new one, never part of
either. If the write fails or is interrupted, the temporary file is removed, the error is raised
again, and the previous file is left as it was.

If the process is killed outright during a write, a temporary file named like
`.profile.json.*.tmp` can be left beside the Profile. It is never read, and it is safe to delete.

The replace is atomic, but it is not guaranteed to be durable: after a power loss or an operating
system crash, the file may still hold the previous Profile. That is safe, since an out-of-date
Profile is a miss and is prepared again.

On Linux and macOS every write leaves the file readable and writable by its owner only, replacing
any mode set on it since the last write.

## Several processes or instances on one file

One store serialises its own reads and writes with a lock, so threads and the sync and async paths
of one instance can share it. Nothing coordinates separate stores, separate instances or separate
processes pointed at the same file:

- **Last writer wins.** Two writers do not merge; the file holds whichever Profile was written last.
  A reader still never sees a mix of the two.
- **There is no cross-process lock.**
- **On Windows, a read can fail** with an OS error while another process is replacing the file, and
  a replace can fail while another process has the file open.

Use one writer per file. If several processes need a Profile, give each its own file, or let one
process write and accept the Windows limit above in the others.
