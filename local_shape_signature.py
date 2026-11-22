"""Deterministic contour evidence for isolated transparent assets, not VLM captions.

Dominant angular frequency is not a universal petal counter. Report the spectrum
and abstain for opaque images or unstable support; downstream use remains scoped
to approximately centered planar motifs.
"""
import numpy as np
from PIL import Image

def shape_signature(path):
    im=Image.open(path).convert('RGBA');a=np.asarray(im)[...,3]
    if a.min()==255:return {'status':'abstain','reason':'no isolated alpha foreground'}
    runs=[]
    for threshold in [64,127,192]:
        y,x=np.nonzero(a>threshold)
        if len(x)<100:return {'status':'abstain','reason':'insufficient alpha support'}
        cx,cy=float(x.mean()),float(y.mean());theta=np.arctan2(y-cy,x-cx);radius=np.hypot(x-cx,y-cy)
        bins=np.minimum(719,((theta+np.pi)*720/(2*np.pi)).astype(int))
        profile=np.zeros(720);np.maximum.at(profile,bins,radius)
        missing=profile==0
        if missing.mean()>.05:return {'status':'abstain','reason':'sparse radial support'}
        profile[missing]=np.interp(np.flatnonzero(missing),np.flatnonzero(~missing),profile[~missing],period=720)
        spectrum=abs(np.fft.rfft(profile-profile.mean()))
        orders=sorted(range(2,13),key=lambda k:float(spectrum[k]),reverse=True)
        runs.append({'alpha_threshold':threshold,'dominant_order':orders[0],
            'amplitude_ratio':float(spectrum[orders[0]]/max(1e-8,spectrum[orders[1]])),
            'top_orders':[{'order':k,'amplitude':float(spectrum[k])} for k in orders[:5]],
            'centroid':[cx/im.width,cy/im.height]})
    stable=len({x['dominant_order'] for x in runs})==1 and min(x['amplitude_ratio'] for x in runs)>1.6
    return {'status':'stable_radial_frequency' if stable else 'ambiguous','runs':runs,
        'dominant_order':runs[0]['dominant_order'] if stable else None,
        'scope':'alpha silhouette rotational evidence; not guaranteed petal count for asymmetric, concave or perspective-distorted designs'}
