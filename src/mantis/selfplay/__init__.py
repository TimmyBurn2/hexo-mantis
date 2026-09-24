"""Self-play: the worker pool over the Rust runner, the one inference server, the graph wire
reader and the replay-buffer facade. Nothing here imports `mantis.eval`, `mantis.train` or
`mantis.bots` (collaborators are injected); import the submodules, the package re-exports nothing.
"""
