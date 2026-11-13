#!/bin/sh
# TRAKTOR ML, Colab cut diagnosis (docs/reports/colab_corte_13min_2026-11-13.md). E1 then E0, sessions diag-e1 and
# diag-e0, with the Colab login of the gpuq container. It uses Gabriel's Colab account and up to ~1.5 h of free T4:
# ask him first, and check that the gpuq queue has nothing for Colab. On the Pi:
#   scp this folder to ~/colab_diag, then: cd ~/colab_diag && nohup sh run_diag.sh all > run.out 2>&1 < /dev/null &
# The argument is e1, e0, all (e1 then e0, the default) or e2 (the candidate fix: detached job, long polls).
# E1: 35 min of GPU work inside one long `colab exec` (connection open). E0: the same work detached from the kernel
# and one short `colab exec` per minute, as gpuq does. Read run.out: a VM that survives prints up to "35.0 min".
D=$HOME/colab_diag
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
stamp() { while IFS= read -r l; do echo "$(date -u +%H:%M:%S) $l"; done; }
WHICH=${1:-all}

if [ "$WHICH" = e1 ] || [ "$WHICH" = all ]; then
echo "$(ts) E1 new"
docker exec gpuq colab new -s diag-e1 --gpu T4 2>&1 | stamp
echo "$(ts) E1 exec start"
docker exec -i gpuq timeout 2700 colab exec -s diag-e1 --timeout 2400 < $D/e1.py 2>&1 | stamp
echo "$(ts) E1 exec end"
docker exec gpuq colab sessions 2>&1 | grep -v @ | stamp
docker exec -i gpuq timeout 180 colab exec -s diag-e1 --timeout 60 < $D/read_log.py 2>&1 | stamp
docker exec gpuq colab stop -s diag-e1 2>&1 | stamp
echo "$(ts) E1 END"
fi

if [ "$WHICH" = e0 ] || [ "$WHICH" = all ]; then
echo "$(ts) E0 new"
docker exec gpuq colab new -s diag-e0 --gpu T4 2>&1 | stamp
docker exec -i gpuq timeout 300 colab exec -s diag-e0 --timeout 120 < $D/e0_start.py 2>&1 | stamp
start=$(date +%s); fails=0
while [ $(( $(date +%s) - start )) -lt 2520 ]; do
  sleep 60
  out=$(docker exec -i gpuq timeout 240 colab exec -s diag-e0 --timeout 120 < $D/e0_poll.py 2>&1)
  if echo "$out" | grep -q "E0POLL"; then echo "$out" | grep "E0POLL" | stamp; fails=0
  else fails=$((fails+1)); echo "$out" | grep -v -E "│|╭|╰" | tail -3 | stamp; echo "$(ts) poll failed $fails"; fi
  # Only the CLI's own line means the VM is gone. A traceback of a network timeout shows the CLI's source,
  # which contains "Kernel not found at", so matching "not found" anywhere took a live VM for a lost one.
  if echo "$out" | grep -q -E "^\[colab\] Session '[^']*' (appears to be lost|not found)"; then echo "$(ts) E0 LOST"; break; fi
  if [ $fails -ge 6 ]; then echo "$(ts) E0 six failed polls"; break; fi
done
docker exec gpuq colab sessions 2>&1 | grep -v @ | stamp
echo 'print(open("/content/e0_loop.out").read()[-1500:] or "e0_loop.out empty")' | docker exec -i gpuq timeout 180 colab exec -s diag-e0 --timeout 60 2>&1 | stamp
docker exec gpuq colab stop -s diag-e0 2>&1 | stamp
fi

if [ "$WHICH" = e2 ]; then
echo "$(ts) E2 new"
docker exec gpuq colab new -s diag-e2 --gpu T4 2>&1 | stamp
docker exec -i gpuq timeout 300 colab exec -s diag-e2 --timeout 120 < $D/e0_start.py 2>&1 | stamp
start=$(date +%s); fails=0
while [ $(( $(date +%s) - start )) -lt 2520 ]; do
  out=$(docker exec -i gpuq timeout 900 colab exec -s diag-e2 --timeout 700 < $D/e2_poll.py 2>&1)
  if echo "$out" | grep -q "E2POLL"; then echo "$out" | grep -E "E2POLL|E2 hold" | tail -2 | stamp; fails=0
  else fails=$((fails+1)); echo "$out" | grep -v -E "│|╭|╰" | tail -3 | stamp; echo "$(ts) poll failed $fails"; sleep 30; fi
  if echo "$out" | grep -q -E "^\[colab\] Session '[^']*' (appears to be lost|not found)"; then echo "$(ts) E2 LOST"; break; fi
  if echo "$out" | grep -q "E0 DONE"; then echo "$(ts) E2 job done"; break; fi
  if [ $fails -ge 6 ]; then echo "$(ts) E2 six failed polls"; break; fi
done
docker exec gpuq colab sessions 2>&1 | grep -v @ | stamp
docker exec gpuq colab stop -s diag-e2 2>&1 | stamp
fi
echo "$(ts) ALL DONE"
