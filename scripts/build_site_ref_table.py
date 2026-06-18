"""Build the shipped GPCRdb generic-numbering + segment table (maintenance tool).

Scope: all human wt GPCR receptors (future-proof for the dominant case) + any
non-human UniProt accession present in the local corpus (current orthologs).
Output: {accession: {"c": class, "e": entry_name, "r": {seqnum: [x_label, segment, aa]}}}
gzipped JSON. Keyed by UniProt accession (what enriched/RCSB-align provides).
Requires the local GPCRdb DB (gpcrdb-db). Run once; ship the artifact.
"""
import json, gzip, glob, subprocess, os

def psql(sql):
    return subprocess.run(["docker","exec","gpcrdb-db","psql","-U","protwis","-d","protwis","-tA","-F","\t","-c",sql],
                          capture_output=True, text=True).stdout

# 1. corpus non-human accessions (from enriched GPCR uniprots)
base="/Users/nht435/GitHub/GPCR-annotation-tools-collab/localdata/corpus_2026-06-01"
corpus_accs=set()
for ep in glob.glob(f"{base}/*/enriched/*.json"):
    try: raw=json.load(open(ep))
    except: continue
    e=(raw.get("data") or {}).get("entry") or raw
    for ent in (e.get("polymer_entities") or []):
        for u in (ent.get("uniprots") or []):
            if u.get("gpcrdb_entry_name_slug"):
                acc=(u.get("rcsb_id") or "").strip()
                if acc: corpus_accs.add(acc)
print(f"corpus GPCR accessions: {len(corpus_accs)}")

in_list = ",".join("'%s'" % a.replace("'","") for a in corpus_accs) or "''"
sql=f"""SELECT p.accession, p.entry_name, split_part(pf.slug,'_',1) AS cls,
       r.sequence_number, COALESCE(gn.label,''), COALESCE(ps.slug,''), r.amino_acid
FROM residue r
JOIN protein_conformation pc ON pc.id=r.protein_conformation_id
JOIN protein p ON p.id=pc.protein_id
JOIN protein_family pf ON pf.id=p.family_id
JOIN protein_sequence_type st ON st.id=p.sequence_type_id AND st.slug='wt'
JOIN species sp ON sp.id=p.species_id
JOIN protein_segment ps ON ps.id=r.protein_segment_id
LEFT JOIN residue_generic_number gn ON gn.id=r.generic_number_id
     AND gn.scheme_id=(SELECT id FROM residue_generic_numbering_scheme WHERE slug='gpcrdb')
WHERE split_part(pf.slug,'_',1) IN ('001','002','003','004','005','006','007','008','009','010')
  AND (sp.latin_name='Homo sapiens' OR p.accession IN ({in_list}))
ORDER BY p.accession, r.sequence_number;"""

table={}
n=0
for line in psql(sql).strip().split("\n"):
    p=line.split("\t")
    if len(p)<7 or not p[0]: continue
    acc,entry,cls,seqn,xlab,seg,aa=p[0],p[1],p[2],p[3],p[4],p[5],p[6]
    rec=table.setdefault(acc,{"c":cls,"e":entry,"r":{}})
    rec["r"][seqn]=[xlab or None, seg or None, aa]
    n+=1

out="/tmp/gpcrdb_generic_numbers.json.gz"
with gzip.open(out,"wt",encoding="utf-8") as f:
    json.dump(table,f,separators=(",",":"))
print(f"receptors: {len(table)} | residue rows: {n} | gz size: {os.path.getsize(out)/1e6:.2f} MB")
# spot check
for acc in ("P07550","Q9NYV8","P41180"):
    if acc in table:
        r=table[acc]["r"]
        landmarks={k:v for k,v in r.items() if v[0] in ("3x32","2x50","6x48","7x39")}
        print(f"  {acc} {table[acc]['e']} class {table[acc]['c']}: {len(r)} residues; landmarks {landmarks}")
