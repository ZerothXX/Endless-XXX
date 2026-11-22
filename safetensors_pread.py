"""Read exact, bounded safetensors byte ranges; no whole-shard torch storage.

Used only by the diagnostic backend on this Windows host. No dtype changes,
weight mutation or model algebra changes. Source hashes remain unchanged.
"""
import json,math,hashlib,struct
from pathlib import Path

class PreadSafeOpen:
 def __init__(self,filename,framework='pt',device='cpu',backend=None):
  assert framework=='pt' and str(device)=='cpu'
  self.path=Path(filename)
  with self.path.open('rb') as f:
   self.header_length=struct.unpack('<Q',f.read(8))[0]
   assert self.header_length<100_000_000
   self.header=json.loads(f.read(self.header_length))
  self.offset=8+self.header_length
 def keys(self):return [k for k in self.header if k!='__metadata__']
 def metadata(self):return self.header.get('__metadata__',{})
 def get_slice(self,name):
  shape=self.header[name]['shape'];dtype=self.header[name]['dtype'];owner=self
  class Slice:
   def get_shape(self):return shape
   def get_dtype(self):return dtype
   def __getitem__(self,index):return owner.get_tensor(name)[index]
  return Slice()
 def raw(self,name):
  lo,hi=self.header[name]['data_offsets'];assert 0<=lo<=hi<=self.path.stat().st_size-self.offset
  with self.path.open('rb') as f:f.seek(self.offset+lo);data=f.read(hi-lo)
  if len(data)!=hi-lo:raise IOError('Short safetensors tensor read')
  return data
 def get_tensor(self,name):
  import torch
  meta=self.header[name];types={'BF16':torch.bfloat16,'F16':torch.float16,'F32':torch.float32,'F64':torch.float64,
   'I64':torch.int64,'I32':torch.int32,'I16':torch.int16,'I8':torch.int8,'U8':torch.uint8,'BOOL':torch.bool}
  dtype=types[meta['dtype']];lo,hi=meta['data_offsets']
  assert 0<=lo<=hi<=self.path.stat().st_size-self.offset
  data=bytearray(hi-lo)
  with self.path.open('rb') as f:
   f.seek(self.offset+lo);assert f.readinto(data)==len(data),'Short safetensors tensor read'
  assert len(data)==math.prod(meta['shape'])*torch.empty((),dtype=dtype).element_size()
  if not data:return torch.empty(meta['shape'],dtype=dtype)
  return torch.frombuffer(data,dtype=dtype).reshape(meta['shape'])
 def __enter__(self):return self
 def __exit__(self,*args):pass

if __name__=='__main__':
 import sys,time,torch
 from safetensors import safe_open
 p=Path(sys.argv[1]);f=PreadSafeOpen(p);rows=[]
 for name in f.keys()[:3]:
  start=time.monotonic();t=f.get_tensor(name)
  raw=t.view(torch.uint8).reshape(-1).numpy().tobytes()
  assert raw==f.raw(name)
  rows.append({'name':name,'shape':list(t.shape),'dtype':str(t.dtype),'sha256':hashlib.sha256(raw).hexdigest(),'seconds':time.monotonic()-start})
 # A small independently serialized file checks parser/layout parity without
 # reproducing the host's stalled large-shard storage path.
 from safetensors.torch import save_file
 test=Path(__file__).parent/'pread_parity.safetensors'
 sample={d:torch.arange(24,dtype=torch.float32).reshape(2,3,4).to(getattr(torch,d)) for d in ['bfloat16','float16','float32','int64']}
 save_file(sample,test)
 with safe_open(test,framework='pt',device='cpu') as reference:
  mine=PreadSafeOpen(test)
  for name in reference.keys():
   a=reference.get_tensor(name);b=mine.get_tensor(name)
   assert a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
 report={'source':str(p.resolve()),'sampled_raw_tensor_roundtrips':rows,'independently_serialized_dtype_parity':['BF16','F16','F32','I64'],
  'scope':'bounded exact byte reads and parser parity; not a model quality assertion'}
 (Path(__file__).parent/'pread_verification.json').write_text(json.dumps(report,indent=2),encoding='utf8')
 print(json.dumps(report,indent=2))
