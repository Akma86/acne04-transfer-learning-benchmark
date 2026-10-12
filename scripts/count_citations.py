import re
from pathlib import Path

paper_dir = Path("docs/paper")
bib_file = paper_dir / "References.bib"

# 1. Parse References.bib
bib_content = bib_file.read_text(encoding="utf-8")
bib_entries = {}
pattern = re.compile(r"@([a-zA-Z]+)\s*\{\s*([^,\s]+)\s*,\s*(.*?)(?=\n@|\Z)", re.DOTALL)
for match in pattern.finditer(bib_content):
    entry_type = match.group(1).lower()
    key = match.group(2).strip()
    body = match.group(3)
    title_m = re.search(r'title\s*=\s*[\"{](.*?)[\"}],?', body, re.IGNORECASE | re.DOTALL)
    title = title_m.group(1).strip() if title_m else "No title"
    title = re.sub(r"[\{\}\s]+", " ", title).strip()
    year_m = re.search(r'year\s*=\s*[\"{]?([0-9]{4})[\"}]?,?', body, re.IGNORECASE)
    year = year_m.group(1).strip() if year_m else "-"
    author_m = re.search(r'author\s*=\s*[\"{](.*?)[\"}],?', body, re.IGNORECASE | re.DOTALL)
    author = author_m.group(1).strip() if author_m else "-"
    author = re.sub(r"[\{\}\s]+", " ", author).strip()
    bib_entries[key] = {"type": entry_type, "title": title, "year": year, "author": author}

# 2. Parse all \cite commands in .tex files
tex_files = sorted(paper_dir.glob("*.tex"))
cited_by_file = {}
all_cited = set()
cite_counts = {}

cite_regex = re.compile(r"\\cite[a-zA-Z]*\{([^}]+)\}")

for tf in tex_files:
    content = tf.read_text(encoding="utf-8")
    # remove comments
    clean_lines = [l for l in content.splitlines() if not l.strip().startswith("%")]
    clean_content = "\n".join(clean_lines)
    
    matches = cite_regex.findall(clean_content)
    file_keys = []
    for m in matches:
        keys = [k.strip() for k in m.split(",") if k.strip()]
        for k in keys:
            file_keys.append(k)
            all_cited.add(k)
            cite_counts[k] = cite_counts.get(k, 0) + 1
    if file_keys:
        cited_by_file[tf.name] = file_keys

print(f"=== RINGKASAN SITASI PAPER ===")
print(f"Total entri yang tersedia di References.bib : {len(bib_entries)}")
print(f"Total kunci sitasi unik yang SUDAH DIPAKAI  : {len(all_cited)}")
print(f"Total pemanggilan sitasi di seluruh naskah   : {sum(cite_counts.values())}")
print()

print("=== DISTRIBUSI PENGGUNAAN SITASI PER BAB/FILE ===")
for fname, keys in cited_by_file.items():
    print(f"- {fname:25s}: {len(keys):2d} sitasi ({len(set(keys)):2d} kunci unik)")
print()

print(f"=== DAFTAR {len(all_cited)} SITASI YANG SUDAH DIPAKAI DI PAPER ===")
missing_in_bib = []
for idx, k in enumerate(sorted(all_cited), 1):
    count = cite_counts[k]
    if k in bib_entries:
        info = bib_entries[k]
        first_author = info['author'].split(' and ')[0] if ' and ' in info['author'] else info['author']
        print(f"{idx:2d}. [{k}] ({count}x) | {first_author} ({info['year']}) - \"{info['title']}\"")
    else:
        print(f"{idx:2d}. [WARNING: TIDAK ADA DI BIB] [{k}] ({count}x)")
        missing_in_bib.append(k)

if missing_in_bib:
    print(f"\n[PERINGATAN] Ada {len(missing_in_bib)} sitasi yang dipanggil di .tex tapi belum ada di References.bib: {missing_in_bib}")
else:
    print("\n[OK] Semua sitasi yang dipakai di naskah SUDAH ADA di References.bib!")

unused_in_bib = set(bib_entries.keys()) - all_cited
print(f"\n=== ENTRI DI REFERENCES.BIB YANG BELUM DIPAKAI ({len(unused_in_bib)} entri) ===")
for idx, k in enumerate(sorted(unused_in_bib), 1):
    info = bib_entries[k]
    first_author = info['author'].split(' and ')[0] if ' and ' in info['author'] else info['author']
    print(f"{idx:2d}. [{k}] | {first_author} ({info['year']}) - \"{info['title']}\"")
