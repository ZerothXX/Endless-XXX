"""Deterministic sRGB/D65 CIELAB clustering using NumPy, no model/service."""
import numpy as np
M=np.array([[.4124564,.3575761,.1804375],[.2126729,.7151522,.0721750],[.0193339,.1191920,.9503041]])
WHITE=np.array([.95047,1.,1.08883])

def rgb2lab(rgb):
    rgb=np.asarray(rgb);linear=np.where(rgb>.04045,((rgb+.055)/1.055)**2.4,rgb/12.92)
    t=(linear@M.T)/WHITE;d=6/29
    f=np.where(t>d**3,np.cbrt(t),t/(3*d*d)+4/29)
    return np.stack([116*f[...,1]-16,500*(f[...,0]-f[...,1]),200*(f[...,1]-f[...,2])],axis=-1)

def lab2rgb(lab):
    lab=np.asarray(lab);y=(lab[...,0]+16)/116
    f=np.stack([y+lab[...,1]/500,y,y-lab[...,2]/200],axis=-1);d=6/29
    xyz=np.where(f>d,f**3,3*d*d*(f-4/29))*WHITE
    linear=xyz@np.linalg.inv(M).T
    return np.clip(np.where(linear>.0031308,1.055*np.maximum(linear,0)**(1/2.4)-.055,12.92*linear),0,1)

def cluster_colors(lab,k=8,seed=17):
    rng=np.random.default_rng(seed);best=None
    for repeat in range(5):
        centers=[lab[rng.integers(len(lab))]]
        for _ in range(1,k):
            distance=((lab[:,None,:]-np.array(centers)[None,:,:])**2).sum(2).min(1)
            centers.append(lab[rng.choice(len(lab),p=distance/distance.sum())] if distance.sum()>0 else lab[0])
        centers=np.array(centers)
        for _ in range(100):
            distance=((lab[:,None,:]-centers[None,:,:])**2).sum(2);labels=distance.argmin(1)
            updated=np.array([lab[labels==i].mean(0) if (labels==i).any() else centers[i] for i in range(k)])
            if np.max(np.abs(updated-centers))<1e-4:break
            centers=updated
        distance=((lab[:,None,:]-centers[None,:,:])**2).sum(2);labels=distance.argmin(1)
        cost=float(distance[np.arange(len(lab)),labels].sum())
        if best is None or cost<best[0]:best=(cost,centers,labels)
    return best[1],best[2]
