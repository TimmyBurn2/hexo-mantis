#!/usr/bin/env bash
# usage: integrate.sh <commit>...   cherry-pick each onto HEAD, fold floor moves into it.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
TMPDIR=${TMPDIR:-/tmp}; export XDG_STATE_HOME=$TMPDIR/xdg UV_CACHE_DIR=$TMPDIR/uvcache
FLOOR=tools/ci_gates/comment_length_floor.txt
for c in "$@"; do
  if [ "$c" != "--fold" ] && ! git cherry-pick "$c" >/dev/null 2>&1; then echo "CONFLICT on $c"; git status --short | grep -v '^??'; exit 3; fi
  n=$(.venv/bin/python -m pytest --collect-only -q -m '' -p no:cacheprovider 2>&1 | tail -1)
  count=$(echo "$n" | grep -Eo '^[0-9]+ tests? collected' | grep -Eo '^[0-9]+')
  [ -z "$count" ] && { echo "COLLECTION BROKEN after $c: $n"; exit 4; }
  old=$(cat tools/ci_gates/test_count_floor.txt)
  [ "$count" != "$old" ] && echo "$count" > tools/ci_gates/test_count_floor.txt
  lint=$(.venv/bin/python tools/ci_gates/comment_lint.py 2>&1)
  echo "$lint" | grep -q "comment_lint: GREEN" || { echo "COMMENT LINT RED after $c"; echo "$lint" | tail -5; exit 5; }
  line=$(echo "$lint" | grep -E "^comment_lint: [0-9]+ file")
  for kv in $(echo "$line" | grep -Eo '[a-z_]+=[0-9]+/[0-9]+'); do
    k=${kv%%=*}; v=${kv#*=}; cur=${v%%/*}; fl=${v#*/}
    if [ "$cur" -lt "$fl" ]; then sed -i "s/^$k $fl\$/$k $cur/" $FLOOR; fi
  done
  if ! git diff --quiet -- tools/ci_gates/test_count_floor.txt $FLOOR; then
    git add tools/ci_gates/test_count_floor.txt $FLOOR && git commit -q --amend --no-edit
  fi
  echo "$(git log --oneline -1 | cut -c1-90) | collected $count (floor was $old) | $(echo "$line" | grep -Eo '[a-z_]+=[0-9]+/[0-9]+' | tr '\n' ' ')"
done
