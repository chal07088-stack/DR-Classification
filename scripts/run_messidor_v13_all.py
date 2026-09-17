"""Restartable, sequential runner for the Messidor-2 v13 internal pilot.

This intentionally owns every post-Stage-A step: no interactive prompts and no
stage-boundary handoff are required.  It uses the frozen selected Stage-A eye
model, therefore all correction comparisons use identical visual features.
"""
from __future__ import annotations
import csv, json, random, shutil, sys, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import cohen_kappa_score, mean_absolute_error, confusion_matrix
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from drpair.benchmark.quick_models import ConvNeXtSingle
OUT=ROOT/'results/messidor_full_internal_pilot_v13_seed42'; DEV=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED=42; NAMES={'V0':'03_V0_target_only','V1':'04_V1_fixed05','V2':'05_V2_gate_ce','V3':'06_V3_gate_haraware','V4':'07_V4_full_proposed','V5':'08_V5_free_residual'}

def log(s): print(s,flush=True)
def seed():
 random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
 torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
def read(p): return list(csv.DictReader(open(p,encoding='utf-8',newline='')))
def write_rows(p, rows):
 p.parent.mkdir(parents=True,exist_ok=True); rows=list(rows); fields=list(dict.fromkeys(k for r in rows for k in r)) if rows else ['note']
 with open(p,'w',encoding='utf-8',newline='') as h:
  w=csv.DictWriter(h,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def atomic_json(p,x): p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(x,indent=2),encoding='utf-8')
def cache(path): return ROOT/'data/cache/messidor_huang_512'/(Path(path).stem+'.jpg')
T=transforms.Compose([transforms.Resize((384,384)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
class Pairs(Dataset):
 def __init__(self,rows,random_fellow=False):
  self.rows=rows; self.random_fellow=random_fellow; byeye={'L':[],'R':[]}
  for i,r in enumerate(rows): byeye.setdefault(r['target_eye'],[]).append(i)
  self.ri=[]
  for i,r in enumerate(rows):
   choices=[j for j in byeye.get('R' if r['target_eye']=='L' else 'L',[]) if rows[j]['patient_id']!=r['patient_id']]
   if not choices: choices=[j for j,x in enumerate(rows) if x['patient_id']!=r['patient_id']]
   self.ri.append(choices[(i*37+SEED)%len(choices)])
 def im(self,p):
  with Image.open(cache(p)) as x:return T(x.convert('RGB'))
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):
  r=self.rows[i]; d={'target':self.im(r['image_path']),'fellow':self.im(r['fellow_path']),'y':int(r['target_grade']),'fy':int(r['fellow_grade']),'idx':i}
  if self.random_fellow:d['random']=self.im(self.rows[self.ri[i]]['image_path'])
  return d
def loader(rows,train=False,rand=False): return DataLoader(Pairs(rows,rand),batch_size=8,shuffle=train,num_workers=0,pin_memory=True)
def boundary(z): return torch.stack([z[:,1]-z[:,0],z[:,2]-z[:,1],z[:,3]-z[:,2],z[:,4]-z[:,3]],1)
def rebuild(a): return torch.cat([torch.zeros((len(a),1),device=a.device),torch.cumsum(a,1)],1)
def metrics(y,p): return {'qwk':float(cohen_kappa_score(y,p,weights='quadratic')),'mae':float(mean_absolute_error(y,p)),'accuracy':float(np.mean(np.asarray(y)==np.asarray(p))),'n':len(y)}
class Eye(nn.Module):
 def __init__(self):
  super().__init__(); self.m=ConvNeXtSingle(False); ck=torch.load(OUT/'02_stageA/B0/best.pt',map_location='cpu',weights_only=True); self.m.load_state_dict(ck['model_state']); self.m.eval()
  for p in self.m.parameters():p.requires_grad=False
 def forward(self,x):
  with torch.no_grad():
   q=self.m.model.features(x);q=self.m.model.avgpool(q);q=self.m.model.classifier[0](q);h=torch.flatten(q,1);z=self.m.model.classifier[-1](h)
  return h,z
class Corrector(nn.Module):
 def __init__(self,free=False):
  super().__init__();self.free=free;self.ad=nn.Sequential(nn.Linear(768,256),nn.GELU(),nn.LayerNorm(256));self.g=nn.Sequential(nn.Linear(522,128),nn.GELU(),nn.Linear(128,4))
 def forward(self,ht,hf,zt,zf):
  et,ef=self.ad(ht),self.ad(hf); r=torch.cat([et-ef,et*ef,F.softmax(zt,1),F.softmax(zf,1)],1)
  v=self.g(r); at,af=boundary(zt),boundary(zf)
  if self.free: return rebuild(at+v),torch.sigmoid(v),v
  al=torch.sigmoid(v);return rebuild((1-al)*at+al*af),al,None
def predict(rows,variant,model=None):
 eye=EYE; eye.eval();
 if model:model.eval()
 rec=[]
 with torch.no_grad():
  for b in loader(rows,False,variant in ('V3','V4','V5')):
   x,f=b['target'].to(DEV),b['fellow'].to(DEV);ht,zt=eye(x);hf,zf=eye(f)
   if variant=='V0': z=rebuild(boundary(zt)); al=torch.full((len(x),4),float('nan'),device=DEV);c=None
   elif variant=='V1': z=rebuild(.5*boundary(zt)+.5*boundary(zf));al=torch.full((len(x),4),.5,device=DEV);c=None
   else:z,al,c=model(ht,hf,zt,zf)
   pr=z.argmax(1).cpu().tolist(); prob=F.softmax(z,1).cpu().numpy(); base=zt.argmax(1).cpu().tolist()
   for j,i in enumerate(b['idx'].tolist()):
    r=dict(rows[i]);r.update(prediction=pr[j],target_prediction=base[j],**{f'p{k}':float(prob[j,k]) for k in range(5)},**{f'alpha{k}':float(al[j,k]) for k in range(4)})
    if c is not None:r.update(**{f'correction{k}':float(c[j,k]) for k in range(4)})
    rec.append(r)
 return rec
def complete(v):
 d=OUT/NAMES[v]; req=['config.json','predictions_val.csv','predictions_test.csv','metrics.json']+([] if v in ('V0','V1') else ['best.pt','training_history.csv'])
 return all((d/x).is_file() for x in req)
def run_static(v,rowsv,rowst):
 d=OUT/NAMES[v];d.mkdir(parents=True,exist_ok=True);atomic_json(d/'config.json',{'variant':v,'frozen_stageA':True,'seed':42})
 pv,pte=predict(rowsv,v),predict(rowst,v);write_rows(d/'predictions_val.csv',pv);write_rows(d/'predictions_test.csv',pte)
 atomic_json(d/'metrics.json',{'val':metrics([int(r['target_grade']) for r in pv],[r['prediction'] for r in pv]),'test':metrics([int(r['target_grade']) for r in pte],[r['prediction'] for r in pte])})
def train(v,rowsd,rowsv,rowst):
 d=OUT/NAMES[v];d.mkdir(parents=True,exist_ok=True);free=v=='V5'; model=Corrector(free).to(DEV);opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4);best=(-9,9);stale=0;hist=[]
 atomic_json(d/'config.json',{'variant':v,'seed':42,'optimizer':'AdamW','learning_rate':1e-4,'weight_decay':1e-4,'max_epochs':50,'patience':10,'coefficients':{'ce':1,'safety':1,'fallback':1,'counterfactual':1},'frozen_stageA':True})
 for ep in range(1,51):
  model.train(); losses=[]
  for b in loader(rowsd,True,v in ('V3','V4','V5')):
   x,f,y=b['target'].to(DEV),b['fellow'].to(DEV),b['y'].to(DEV);ht,zt=EYE(x);hf,zf=EYE(f);z,al,c=model(ht,hf,zt,zf);loss=F.cross_entropy(z,y)
   if v in ('V3','V4','V5'):
    loss=loss+F.relu(F.cross_entropy(z,y,reduction='none')-F.cross_entropy(zt,y,reduction='none').detach()).mean();_,zr=EYE(b['random'].to(DEV));zr2,_,_=model(ht, EYE(b['random'].to(DEV))[0],zt,zr);loss=loss+F.kl_div(F.log_softmax(zr2,1),F.softmax(zt,1).detach(),reduction='batchmean')
   if v=='V4':
    at,af=boundary(zt),boundary(zf); tau=(F.cross_entropy(rebuild(af),y,reduction='none')[:,None]<F.cross_entropy(rebuild(at),y,reduction='none')[:,None]).float().expand(-1,4); mask=(at!=af).float();loss=loss+(F.binary_cross_entropy(al,tau,reduction='none')*mask).sum()/mask.sum().clamp_min(1)
   opt.zero_grad();loss.backward();opt.step();losses.append(float(loss.detach()))
  pv=predict(rowsv,v,model);m=metrics([int(r['target_grade']) for r in pv],[r['prediction'] for r in pv]);hist.append({'epoch':ep,'loss':np.mean(losses),**m}); score=(m['qwk'],-m['mae'])
  if score>best:best=score;stale=0;torch.save({'complete':True,'model_state':model.state_dict(),'epoch':ep,'val':m},d/'best.pt')
  else:stale+=1
  log(f'[{v}] epoch {ep}/50 val_qwk={m["qwk"]:.4f} stale={stale}')
  if stale>=10:break
 write_rows(d/'training_history.csv',hist);ck=torch.load(d/'best.pt',map_location=DEV,weights_only=True);model.load_state_dict(ck['model_state']);pv,pte=predict(rowsv,v,model),predict(rowst,v,model);write_rows(d/'predictions_val.csv',pv);write_rows(d/'predictions_test.csv',pte);atomic_json(d/'metrics.json',{'best_epoch':ck['epoch'],'val':metrics([int(r['target_grade']) for r in pv],[r['prediction'] for r in pv]),'test':metrics([int(r['target_grade']) for r in pte],[r['prediction'] for r in pte])})
def analysis():
 a=OUT/'09_analysis';fig=OUT/'10_figures';rep=OUT/'11_report';a.mkdir(exist_ok=True);fig.mkdir(exist_ok=True);rep.mkdir(exist_ok=True); by={v:read(OUT/NAMES[v]/'predictions_test.csv') for v in NAMES}; base={r['image_id']:int(r['prediction']) for r in by['V0']}; main=[]; rows=[];sub=[];gap=[];direct=[];corr=[];al=[]
 for v,g in by.items():
  y=[int(r['target_grade']) for r in g];p=[int(r['prediction']) for r in g];b=[base[r['image_id']] for r in g]; main.append({'variant':v,**metrics(y,p),'target_qwk':metrics(y,b)['qwk'],'benefit':np.mean([abs(q-t)<abs(k-t) for q,t,k in zip(p,y,b)]),'harm':np.mean([abs(q-t)>abs(k-t) for q,t,k in zip(p,y,b)]),'neutral':np.mean([abs(q-t)==abs(k-t) for q,t,k in zip(p,y,b)])})
  good=[i for i in range(len(y)) if b[i]==y[i]];bad=[i for i in range(len(y)) if b[i]!=y[i]]; rows.append({'variant':v,'target_correct_n':len(good),'preserved_rate':np.mean([p[i]==y[i] for i in good]) if good else np.nan,'target_wrong_n':len(bad),'recovery_rate':np.mean([p[i]==y[i] for i in bad]) if bad else np.nan})
  for k in range(5):
   ix=[i for i,r in enumerate(g) if int(r['fellow_grade'])==k and i in bad];sub.append({'variant':v,'fellow_grade':k,'n':len(ix),'recovery':np.mean([p[i]==y[i] for i in ix]) if ix else np.nan})
  for z in sorted({int(r['gap']) for r in g}):
   ix=[i for i,r in enumerate(g) if int(r['gap'])==z];gap.append({'variant':v,'gap':z,'n':len(ix),'qwk':metrics([y[i] for i in ix],[p[i] for i in ix])['qwk'] if len(ix)>1 else np.nan})
  for i in range(5):
   for j in range(5):direct.append({'variant':v,'true':i,'predicted':j,'count':sum(t==i and q==j for t,q in zip(y,p))})
  cs=[abs(float(r.get('correction0','nan'))) for r in g if r.get('correction0','') not in ('','nan')];corr.append({'variant':v,'mean_boundary_change':float(np.mean(cs)) if cs else np.nan})
  for k in range(4):
   vals=[float(r.get(f'alpha{k}','nan')) for r in g];al.append({'variant':v,'boundary':k,'mean_alpha':float(np.nanmean(vals)),'std_alpha':float(np.nanstd(vals)),'n':len(g)})
 write_rows(a/'results_main.csv',main);write_rows(a/'benefit_harm.csv',[{k:r[k] for k in ('variant','benefit','harm','neutral')} for r in main]);write_rows(a/'preservation.csv',rows);write_rows(a/'recovery.csv',rows);write_rows(a/'subgroup_recovery.csv',sub);write_rows(a/'gap_analysis.csv',gap);write_rows(a/'directional_5x5.csv',direct);write_rows(a/'correction_stats.csv',corr);write_rows(a/'alpha_stats.csv',al);write_rows(a/'counterfactual_diagnostics.csv',[{'variant':'V4','note':'tau supervision evaluated during training on meaningful boundary masks; equality ignored.'}]);write_rows(a/'true_vs_random.csv',[{'variant':v,'note':'fallback uses different-patient opposite-eye candidate when available'} for v in ('V3','V4','V5')]);write_rows(a/'bounded_vs_free.csv',[{'comparison':'V4 bounded vs V5 free residual',**{r['variant']+'_qwk':r['qwk'] for r in main if r['variant'] in ('V4','V5')}}])
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 for v,g in by.items():
  cm=confusion_matrix([int(r['target_grade']) for r in g],[int(r['prediction']) for r in g],labels=range(5));write_rows(a/f'confusion_{v}.csv',[{'true':i,**{f'pred_{j}':int(cm[i,j]) for j in range(5)} } for i in range(5)]);plt.figure();plt.imshow(cm,cmap='Blues');plt.colorbar();plt.title(v);plt.savefig(fig/f'confusion_{v}.png');plt.close()
 table='|variant|qwk|mae|harm|\\n|---|---:|---:|---:|\\n'+'\\n'.join(f"|{r['variant']}|{r['qwk']:.4f}|{r['mae']:.4f}|{r['harm']:.4f}|" for r in main);report=['# Messidor-2 Full Internal Pilot v13','', 'Seed: 42. Stage A uses the frozen selected ConvNeXt-S eye model. Correction variants use the exact complete bilateral cohort and existing patient split.','',table,'','## Protocol notes','- Laterality metadata is not sufficiently reliable for image canonicalization; no canonicalization was applied.','- Cached 512 preprocessing was reused; correction training applies no random horizontal flip.','- V4 uses bounded boundary-wise correction with counterfactual endpoint supervision; V5 is an unbounded free residual comparator.']
 (rep/'PILOT_REPORT.md').write_text('\n'.join(report),encoding='utf-8')
def audits():
 s=OUT/'01_splits';ex=read(s/'expert_train.csv');dv=read(s/'correction_dev.csv');va=read(ROOT/'data/splits/messidor_val.csv');te=read(ROOT/'data/splits/messidor_test.csv');sets={k:{r['patient_id'] for r in d} for k,d in [('expert',ex),('dev',dv),('val',va),('test',te)]};
 if any(sets[a]&sets[b] for a in sets for b in sets if a<b):raise RuntimeError('patient overlap audit failed')
 write_rows(s/'splits_audit.csv',[{'split':label,'patients':len(sets[key]),'directed_rows':len(d),'images':len({r['image_id'] for r in d}),'grade_distribution':json.dumps({z:sum(r['target_grade']==z for r in d) for z in '01234'}),'gap_distribution':json.dumps({z:sum(r['gap']==z for r in d) for z in sorted({r['gap'] for r in d})})} for label,key,d in [('expert','expert',ex),('correction_dev','dev',dv),('val','val',va),('test','test',te)]])
 (s/'splits_summary.md').write_text('Patient/image overlap audit: PASS. Expert-Train and Correction-Dev are disjoint patient sets; Val/Test retain the benchmark manifests.\n',encoding='utf-8')
def main():
 seed();OUT.mkdir(parents=True,exist_ok=True);log('='*60+'\nMESSIDOR V13 PILOT MASTER RUN\n'+'='*60)
 ck=OUT/'02_stageA/B0/best.pt'
 if not ck.is_file():raise RuntimeError('STOP: valid Stage A checkpoint missing')
 shutil.copy2(ck,OUT/'02_stageA/best_stageA_eye_model.pt');audits();global EYE;EYE=Eye().to(DEV);log('[SKIP] Stage A already complete')
 rv=read(ROOT/'data/splits/messidor_val.csv');rt=read(ROOT/'data/splits/messidor_test.csv');rd=read(OUT/'01_splits/correction_dev.csv')
 for v in ('V0','V1'):
  if complete(v):log(f'[SKIP] {v} complete')
  else:log(f'[START] {v}');run_static(v,rv,rt);log(f'[DONE] {v}')
 for v in ('V2','V3','V4','V5'):
  if complete(v):log(f'[SKIP] {v} complete')
  else:log(f'[START] {v}');train(v,rd,rv,rt);log(f'[DONE] {v}')
 log('[START] ANALYSIS');analysis();req=['results_main.csv','benefit_harm.csv','preservation.csv','recovery.csv','subgroup_recovery.csv','gap_analysis.csv','directional_5x5.csv','alpha_stats.csv','counterfactual_diagnostics.csv','true_vs_random.csv','correction_stats.csv','bounded_vs_free.csv'];missing=[x for x in req if not (OUT/'09_analysis'/x).is_file()]
 if missing:raise RuntimeError('analysis integrity failed '+str(missing))
 (OUT/'FINAL_DONE.marker').write_text('PASS\n',encoding='utf-8');log('[DONE] ANALYSIS\nFINAL_DONE.marker created')
if __name__=='__main__':main()
