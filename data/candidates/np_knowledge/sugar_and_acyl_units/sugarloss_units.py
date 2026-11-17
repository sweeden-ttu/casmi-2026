"""Conservative, stereochemistry-independent natural-product substructure annotations."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('NUMBA_NUM_THREADS','1')
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

FORMULAS={
 'hexose':'C6H10O5','deoxyhexose':'C6H10O4','pentose':'C5H8O4',
 'uronic_acid':'C6H8O6','hex_deoxy_disaccharide':'C12H20O9',
 'malonyl':'C3H2O3','galloyl':'C7H4O4','caffeoyl':'C9H6O3',
 'feruloyl':'C10H8O3','sinapoyl':'C11H10O4','sulfate':'SO3','phosphate':'H3PO4'}
SMARTS={
 'galloyl':'[#6]-[OX2]-[CX3](=[OX1])-c1[cH]c([O;H1,-1])c([O;H1,-1])c([O;H1,-1])[cH]1',
 'caffeoyl':'[#6]-[OX2]-[CX3](=[OX1])-[CH]=[CH]-c1[cH]c([O;H1,-1])c([O;H1,-1])[cH][cH]1',
 'feruloyl':'[#6]-[OX2]-[CX3](=[OX1])-[CH]=[CH]-c1[cH]c([OX2]-[CH3])c([O;H1,-1])[cH][cH]1',
 'sinapoyl':'[#6]-[OX2]-[CX3](=[OX1])-[CH]=[CH]-c1[cH]c([OX2]-[CH3])c([O;H1,-1])c([OX2]-[CH3])[cH]1',
 'malonyl':'[OX2]-[CX3](=[OX1])-[CH2]-[CX3](=[OX1])-[O]',
 'sulfate':'[SX4](=[OX1])(=[OX1])([OX2]-[#6])-[O;H1,-1,$([OX2]-[#6])]',
 'phosphate':'[PX4](=[OX1])([OX2]-[#6])([O;H1,-1,$([OX2]-[#6])])-[O;H1,-1,$([OX2]-[#6])]',
}
QUERIES={k:Chem.MolFromSmarts(v) for k,v in SMARTS.items()}
assert all(QUERIES.values())
UNIT_NAMES=list(FORMULAS)
EXTRA_NAMES=['rutinoside_like','sugar_ring_other','sugar_pyranose','sugar_furanose']

def exact_masses():
 import re
 pt=Chem.GetPeriodicTable()
 return {k:sum(pt.GetMostCommonIsotopeMass(e)*int(n or '1') for e,n in re.findall(r'([A-Z][a-z]?)(\d*)',f)) for k,f in FORMULAS.items()}

def annotate(smiles, details=False):
 counts={k:0 for k in UNIT_NAMES+EXTRA_NAMES}; counts['valid_structure']=0
 m=Chem.MolFromSmiles(str(smiles)) if smiles else None
 if m is None: return counts
 counts['valid_structure']=1
 for k,q in QUERIES.items():
  matches=m.GetSubstructMatches(q,uniquify=True)
  if k=='malonyl':
   # Deduplicate mono-/diesters at the central methylene, require >=1 ester O.
   ids={v[3] for v in matches if any(m.GetAtomWithIdx(v[o]).GetDegree()==2 and any(a.GetAtomicNum()==6 and a.GetIdx()!=v[c] for a in m.GetAtomWithIdx(v[o]).GetNeighbors()) for o,c in ((0,1),(6,4)))}
  else:
   center=0 if k in ('sulfate','phosphate') else 2
   ids={v[center] for v in matches}
  counts[k]=len(ids)
 rings=[]
 for rr in m.GetRingInfo().AtomRings():
  ring=set(rr)
  if len(ring) not in (5,6): continue
  ra=[m.GetAtomWithIdx(i) for i in ring]
  if sum(a.GetAtomicNum()==8 for a in ra)!=1 or any(a.GetAtomicNum() not in (6,8) or a.GetIsAromatic() for a in ra): continue
  if any(m.GetBondBetweenAtoms(rr[j],rr[(j+1)%len(rr)]).GetBondType()!=Chem.BondType.SINGLE for j in range(len(rr))): continue
  carb={a.GetIdx() for a in ra if a.GetAtomicNum()==6}
  # Carbon side chains up to two atoms; no fused rings, aromatic carbons or aglycone via O.
  pending=list(carb); rejected=False
  while pending:
   i=pending.pop()
   for a in m.GetAtomWithIdx(i).GetNeighbors():
    j=a.GetIdx()
    if a.GetAtomicNum()!=6 or j in carb: continue
    if a.IsInRing() and j not in ring: rejected=True; continue
    if a.GetIsAromatic(): rejected=True; continue
    carb.add(j); pending.append(j)
    if len(carb)>6: rejected=True; pending=[]; break
  if rejected: continue
  oxy={a.GetIdx() for i in carb for a in m.GetAtomWithIdx(i).GetNeighbors() if a.GetAtomicNum()==8 and a.GetIdx() not in ring}
  if len(oxy)<3: continue
  ro=next(a.GetIdx() for a in ra if a.GetAtomicNum()==8)
  anomeric={a.GetIdx() for a in m.GetAtomWithIdx(ro).GetNeighbors() if a.GetIdx() in ring}
  links=[]
  for i in anomeric:
   for o in m.GetAtomWithIdx(i).GetNeighbors():
    if o.GetIdx() not in oxy or o.GetDegree()!=2: continue
    for a in o.GetNeighbors():
     if a.GetIdx() in carb or a.GetAtomicNum()!=6: continue
     # Anomeric esters are not O-glycosidic acetals.
     if any(b.GetBondType()==Chem.BondType.DOUBLE and b.GetOtherAtom(a).GetAtomicNum()==8 for b in a.GetBonds()): continue
     links.append((i,o.GetIdx(),a.GetIdx()))
  if not links: continue
  nC=len(carb); nO=len(oxy)
  acid=any(sum(a.GetAtomicNum()==8 for a in m.GetAtomWithIdx(i).GetNeighbors())==2 and any(b.GetBondType()==Chem.BondType.DOUBLE and b.GetOtherAtom(m.GetAtomWithIdx(i)).GetAtomicNum()==8 for b in m.GetAtomWithIdx(i).GetBonds()) for i in carb-ring)
  methyl=any(m.GetAtomWithIdx(i).GetTotalNumHs()==3 for i in carb-ring)
  typ='sugar_ring_other'
  if nC==6 and nO==6 and acid: typ='uronic_acid'
  elif nC==6 and nO==5 and not acid: typ='hexose'
  elif nC==6 and nO==4 and not acid: typ='deoxyhexose'
  elif nC==5 and nO==4 and not acid: typ='pentose'
  counts[typ]+=1
  counts['sugar_pyranose' if len(ring)==6 else 'sugar_furanose']+=1
  rings.append({'ring':ring,'carbons':carb,'type':typ,'links':links,'terminal_methyl':methyl})
 for a in rings:
  if a['type']!='deoxyhexose': continue
  for b in rings:
   if b['type']!='hexose': continue
   links=[x for x in a['links'] if x[2] in b['carbons']]
   if links:
    counts['hex_deoxy_disaccharide']+=1
    if a['terminal_methyl'] and any(x[2] not in b['ring'] and m.GetAtomWithIdx(x[2]).GetTotalNumHs()==2 for x in links): counts['rutinoside_like']+=1
 if details:
  return counts, [{'type':x['type'],'ring':sorted(x['ring']),'carbons':sorted(x['carbons']),'links':x['links']} for x in rings]
 return counts
