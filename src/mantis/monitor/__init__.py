"""HEADLESS ONLY: event emit, producer manifest, alert rules — no display code, ever.

The public seams of the monitor package. Import-light and torch-free by law
(§2 L94, census-pinned O-18/O-19): stdlib + `yaml` + at most `mantis.util`/`mantis.encoding`
— nothing here may import `mantis.train`, `mantis.selfplay` or `mantis.eval`, so the
out-of-process supervisor loads in milliseconds on a box whose GPU has wedged.
"""
