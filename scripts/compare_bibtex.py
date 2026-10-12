import re
from pathlib import Path

asli_path = Path(r"d:\MAIN DATA\Documents\Semester 7\Propo\Citation\Asli.txt")
tambahan_path = Path(r"d:\MAIN DATA\Documents\Semester 7\Propo\Citation\Tambahan.txt")

def parse_bibtex(text):
    text = re.sub(r"^```.*?\n", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n```.*$", "", text, flags=re.MULTILINE)
    
    entries = []
    # Pattern to match bibtex entries: @type{key, ... \n}
    # Matches balanced or multiline entries
    pattern = re.compile(r"(@[a-zA-Z]+\s*\{\s*([^,\s]+)\s*,\s*(.*?)\n\})", re.DOTALL)
    for match in pattern.finditer(text):
        raw_full = match.group(1).strip()
        key = match.group(2).strip()
        body = match.group(3)
        
        # Extract title
        title_m = re.search(r"title\s*=\s*[\"{](.*?)[\"}],?", body, re.IGNORECASE | re.DOTALL)
        title = title_m.group(1).strip() if title_m else ""
        # Clean title for comparison
        title_clean = re.sub(r"[\{\}\s]+", " ", title).strip().lower()
        # Remove trailing commas or extra quotes
        title_clean = title_clean.rstrip('",')
        
        # Extract doi
        doi_m = re.search(r"doi\s*=\s*[\"{](.*?)[\"}],?", body, re.IGNORECASE)
        doi = doi_m.group(1).strip() if doi_m else ""

        entries.append({
            "key": key,
            "title": title,
            "title_clean": title_clean,
            "doi": doi,
            "body": body,
            "raw": raw_full
        })
    return entries

asli_entries = parse_bibtex(asli_path.read_text(encoding="utf-8"))
tambahan_entries = parse_bibtex(tambahan_path.read_text(encoding="utf-8"))

print(f"Total entries in Asli: {len(asli_entries)}")
print(f"Total entries in Tambahan: {len(tambahan_entries)}")

asli_keys = {e["key"] for e in asli_entries}
tambahan_keys = {e["key"] for e in tambahan_entries}

common_keys = asli_keys & tambahan_keys
print(f"\n[DUPLICATE BY KEY] Found {len(common_keys)} identical keys:")
for k in sorted(common_keys):
    print(f"  - {k}")

# Also check duplicate by clean title
asli_by_title = {e["title_clean"]: e for e in asli_entries if e["title_clean"]}
tambahan_by_title = {e["title_clean"]: e for e in tambahan_entries if e["title_clean"]}

common_titles = set(asli_by_title.keys()) & set(tambahan_by_title.keys())
print(f"\n[DUPLICATE BY TITLE] Found {len(common_titles)} matching titles:")
for t in sorted(common_titles):
    print(f"  - '{t[:60]}...'")
    print(f"      Asli key: {asli_by_title[t]['key']} | Tambahan key: {tambahan_by_title[t]['key']}")

# Entries only in Tambahan
truly_new = [
    e for e in tambahan_entries 
    if e["key"] not in asli_keys and e["title_clean"] not in asli_by_title
]
print(f"\n[NEW ENTRIES TO ADD] ({len(truly_new)} entries):")
for e in truly_new:
    print(f"  + [{e['key']}] {e['title']}")

# Entries in Asli that are NOT in Tambahan
asli_only = [
    e for e in asli_entries
    if e["key"] not in tambahan_keys and e["title_clean"] not in tambahan_by_title
]
print(f"\n[ENTRIES UNIQUE TO ASLI] ({len(asli_only)} entries):")
for e in asli_only:
    print(f"  * [{e['key']}] {e['title']}")
