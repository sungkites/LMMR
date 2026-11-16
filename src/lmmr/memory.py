"""Source-preserving nodes and interval merging used by LMMR."""
import collections
import re

STOP=set('a an the of to in on at is are was were be been i me my we our you your it its this that these those and or for with from do did does have has had how what which when where who whom many much please can could would should about as by'.split())

PREFIX='<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'

SUFFIX='<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'

STANDARD='Given a question, retrieve relevant passages that answer the question.'

CONDITIONAL='Given the question and the known evidence, retrieve a different passage that supplies the missing information needed to answer the question. Prefer complementary evidence over repetition of the known evidence.'

SYSTEM='Answer using only the supplied evidence. Return a concise direct answer without explanation. If the evidence is insufficient, say "not enough information".'

def words(t):return[w for w in re.findall(r'[a-z0-9]+',t.lower())if w not in STOP]

def overlap(a,d):
 def units(s):
  w=words(s);return set(w)|{w[i]+' '+w[i+1]for i in range(len(w)-1)}
 aa,dd=units(a),units(d);return len(aa&dd)/(len(dd)+1e-9)

def intervals(spans):
 out=collections.defaultdict(list)
 for s in spans:out[(s['paragraph'],s['sentence'])].append((s['start'],s['end']))
 merged={}
 for k,vs in out.items():
  zz=[]
  for a,b in sorted(vs):
   if zz and a<=zz[-1][1]:zz[-1]=(zz[-1][0],max(zz[-1][1],b))
   else:zz.append((a,b))
  merged[k]=zz
 return merged

def contains(a,b):
 aa=intervals(a['spans']);bb=intervals(b['spans'])
 return all(any(x<=u and y>=v for x,y in aa.get(k,[]))for k,vs in bb.items()for u,v in vs)

def merge(task,nodes):
 groups=intervals([s for n in nodes for s in n['spans']]);bypar=collections.defaultdict(list)
 for(pi,si),vs in sorted(groups.items()):
  p=task['paragraphs'][pi];original=p['sentences'][si].strip()
  for start,end in vs:
   label=f'P{pi}:S{si}'+(''if start==0 and end==len(original)else f'@{start}-{end}')
   bypar[pi].append('['+label+'] '+p['title']+': '+original[start:end])
 return'\n\n'.join('\n'.join(bypar[k])for k in sorted(bypar))

def coverage(task,node,gold):
 vv=intervals(node['spans']);got=set()
 for title,si in gold:
  pi=next(i for i,p in enumerate(task['paragraphs'])if p['title']==title)
  if any(a==0 and b>=len(task['paragraphs'][pi]['sentences'][si].strip())for a,b in vv.get((pi,si),[])):got.add((title,si))
 return got

def build_nodes(task,tok):
 out=[]
 def text(spans):return merge(task,[{'spans':spans}])
 def count(ss):return len(tok.encode(text(ss),add_special_tokens=False))
 def split_sentence(pi,si,limit):
  s=task['paragraphs'][pi]['sentences'][si].strip();bounds=[m.end()for m in re.finditer(r'\S+(?:\s+|$)',s)];start=0;parts=[]
  while start<len(s):
   candidate=[b for b in bounds if b>start];lo=0;hi=len(candidate)-1;best=None
   while lo<=hi:
    mid=(lo+hi)//2;sp={'paragraph':pi,'sentence':si,'start':start,'end':candidate[mid]}
    if count([sp])<=limit:best=sp;lo=mid+1
    else:hi=mid-1
   if best is None:
    # A single extreme word is split by characters; no silent truncation.
    end=start+1
    while end<len(s)and count([{'paragraph':pi,'sentence':si,'start':start,'end':end+1}])<=limit:end+=1
    best={'paragraph':pi,'sentence':si,'start':start,'end':end}
   parts.append(best);start=best['end']
  return parts
 def commit(pi,ss,grain):
  if not ss:return
  value=text(ss);n={'id':f'P{pi}:'+('C'if grain=='coarse'else'F')+str(sum(v['paragraph']==pi and v['grain']==grain for v in out)),
   'paragraph':pi,'title':task['paragraphs'][pi]['title'],'grain':grain,'spans':ss,'text':value,'source_ids':sorted({f"P{s['paragraph']}:S{s['sentence']}"for s in ss}),'tokens':count(ss)}
  if grain=='fine'and any(v['text']==value and v['paragraph']==pi for v in out):return
  out.append(n)
 for pi,p in enumerate(task['paragraphs']):
  current=[];nonempty=[si for si,s in enumerate(p['sentences'])if s.strip()]
  for si in nonempty:
   span={'paragraph':pi,'sentence':si,'start':0,'end':len(p['sentences'][si].strip())}
   if count([span])>192:
    commit(pi,current,'coarse');current=[]
    for part in split_sentence(pi,si,192):commit(pi,[part],'coarse')
   elif current and count(current+[span])>192:commit(pi,current,'coarse');current=[span]
   else:current.append(span)
  commit(pi,current,'coarse')
  full=[{'paragraph':pi,'sentence':si,'start':0,'end':len(p['sentences'][si].strip())}for si in nonempty]
  if len(nonempty)>1 or(full and count(full)>128):
   for pos,si in enumerate(nonempty):
    span=full[pos]
    if count([span])>96:
     for part in split_sentence(pi,si,96):commit(pi,[part],'fine')
    else:
     commit(pi,[span],'fine')
     if pos+1<len(full)and full[pos+1]['sentence']==si+1 and count([span,full[pos+1]])<=96:commit(pi,[span,full[pos+1]],'fine')
 assert all(n['tokens']<=(192 if n['grain']=='coarse'else 96)for n in out)
 return out
