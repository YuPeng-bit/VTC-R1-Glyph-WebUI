"""CPU subprocess using the unchanged Video-3D-LLM caption scorer classes."""
import json
import sys

from . import metrics
from vtc_official_eval.caption_eval.bleu.bleu import Bleu
from vtc_official_eval.caption_eval.cider.cider import Cider
from vtc_official_eval.caption_eval.meteor.meteor import Meteor
from vtc_official_eval.caption_eval.rouge.rouge import Rouge


def main():
    payload = json.load(sys.stdin)
    gts, res = payload['gts'], payload['res']
    out = {}
    with metrics._quiet():
        out['CIDEr'] = float(Cider().compute_score(gts, res)[0]) * 100
        for i, score in enumerate(Bleu().compute_score(gts, res)[0], 1):
            out[f'BLEU-{i}'] = float(score) * 100
        out['ROUGE-L'] = float(Rouge().compute_score(gts, res)[0]) * 100
    # Only transport escaping is applied to METEOR's line protocol. No answer
    # truncation, GT-based selection, punctuation/number substitution is used.
    clean = lambda value: ' '.join(str(value).replace('|||', ' ').split())
    meteor = Meteor()
    try:
        with metrics._quiet():
            out['METEOR'] = float(meteor.compute_score(
                {k: [clean(s) for s in v] for k, v in gts.items()},
                {k: [clean(s) for s in v] for k, v in res.items()})[0]) * 100
    finally:
        meteor.meteor_p.kill()
        meteor.meteor_p.wait(timeout=5)
    print(json.dumps(out, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
