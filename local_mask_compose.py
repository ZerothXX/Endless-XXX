"""Composite only automatically grounded edit support onto the original RGBA.

This guarantees bytes outside support, not semantic fidelity or correct placement.
Candidate/source framing must already agree; no hidden registration or pasted
reference art. A guard rejects masks covering implausibly large image fractions.
"""
import argparse
from pathlib import Path
from local_multiref import ROOT,read,write,sha

def compose(source,candidate,regions,label,dest,max_fraction=.15):
    import numpy as np
    from PIL import Image,ImageOps,ImageChops,ImageFilter
    evidence=read(regions)
    if sha(candidate)!=evidence['source_sha256']:raise ValueError('Grounding belongs to another candidate')
    with Image.open(source) as raw:original=ImageOps.exif_transpose(raw).convert('RGBA')
    with Image.open(candidate) as raw:edited=ImageOps.exif_transpose(raw).convert('RGBA')
    support=Image.new('L',edited.size)
    selected=[r for r in evidence['regions'] if r['label']==label and r['detection_score']>=.3 and r['segmentation_score']>=.8]
    if not selected:raise ValueError('No confident grounded edit region; abstaining')
    for region in selected:
        if sha(region['mask'])!=region['mask_sha256']:raise ValueError('Mask changed')
        support=ImageChops.lighter(support,Image.open(region['mask']).convert('L'))
    support=support.resize(original.size,Image.Resampling.NEAREST)
    hard=np.array(support)>0;fraction=float(hard.mean())
    if fraction>max_fraction or fraction<.0001:raise ValueError('Region guard rejected: '+str(fraction))
    # Feather inward only: support never spreads onto protected pixels.
    feather=ImageChops.multiply(support,support.filter(ImageFilter.GaussianBlur(1.0)))
    edited=edited.resize(original.size,Image.Resampling.LANCZOS)
    result=Image.composite(edited,original,feather)
    dest=Path(dest);dest.parent.mkdir(parents=True,exist_ok=True);result.save(dest)
    with Image.open(dest) as saved:actual=np.asarray(saved.convert('RGBA'))
    changed=np.any(actual!=np.asarray(original),axis=-1)
    outside_changes=int(np.count_nonzero(changed&~hard))
    if outside_changes:raise AssertionError('Protected RGBA pixels changed')
    return {'source':str(Path(source).resolve()),'candidate':str(Path(candidate).resolve()),
        'source_sha256':sha(source),'candidate_sha256':sha(candidate),'regions':str(Path(regions).resolve()),
        'grounded_label':label,'mask_fraction':fraction,'outside_rgba_changed_pixels':outside_changes,
        'output':str(dest.resolve()),'sha256':sha(dest),'actual_size':list(original.size),
        'method':'local model output + automatically grounded mask + inward feathered RGBA composition',
        'worker_sha256':sha(__file__),'online_inference':False,'quality_accepted':False}

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--candidate',required=True)
    p.add_argument('--regions',required=True);p.add_argument('--label',required=True);p.add_argument('--output',required=True)
    p.add_argument('--record',required=True);a=p.parse_args()
    write(a.record,compose(a.source,a.candidate,a.regions,a.label,a.output));print(a.output)

if __name__=='__main__':main()
