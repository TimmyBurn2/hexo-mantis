"""Print run10's resolved config (the post-validation model_dump written to a run dir) as sorted JSON."""
import json
import sys

from mantis.config.loader import load_config

cfg = load_config(sys.argv[1])
print(json.dumps(cfg.model_dump(mode="json"), sort_keys=True, indent=1))
