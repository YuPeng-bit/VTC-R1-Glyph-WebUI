"""CPU-only isolated unchanged official METEOR, with line-safe transport."""
import json, sys
from . import metrics
from vtc_official_eval.caption_eval.meteor.meteor import Meteor

payload=json.load(sys.stdin)
# Whitespace normalization prevents embedded newlines being parsed as commands.
# This never selects answer words, shortens reasoning, or looks at GT agreement.
clean=lambda text:' '.join(str(text).replace('|||',' ').split())
scorer=Meteor()
try:
    with metrics._quiet():
        _,per=scorer.compute_score({'case':[clean(r) for r in payload['references']]},
            {'case':[clean(payload['hypothesis'])]})
    print(json.dumps({'score':float(per[0]),'transport':'single-line whitespace encoding'}),flush=True)
finally:
    scorer.meteor_p.kill();scorer.meteor_p.wait(timeout=5)
