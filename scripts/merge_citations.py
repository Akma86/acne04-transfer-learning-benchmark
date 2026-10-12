import re
from pathlib import Path

citation_dir = Path(r"d:\MAIN DATA\Documents\Semester 7\Propo\Citation")
asli_path = citation_dir / "Asli.txt"
tambahan_path = citation_dir / "Tambahan.txt"
output_path = citation_dir / "fr.txt"

def parse_bibtex_robust(text):
    text = re.sub(r"^```.*?\n", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n```.*$", "", text, flags=re.MULTILINE)
    
    entries = []
    i = 0
    n = len(text)
    while i < n:
        at_pos = text.find('@', i)
        if at_pos == -1:
            break
        brace_open = text.find('{', at_pos)
        if brace_open == -1:
            break
        entry_type = text[at_pos+1:brace_open].strip().lower()
        if not entry_type.isalpha():
            i = at_pos + 1
            continue
        depth = 1
        j = brace_open + 1
        while j < n and depth > 0:
            if text[j] == '{':
                depth += 1
            elif text[j] == '}':
                depth -= 1
            j += 1
        if depth == 0:
            raw_entry = text[at_pos:j].strip()
            comma_pos = raw_entry.find(',')
            if comma_pos != -1:
                key = raw_entry[len(entry_type)+2:comma_pos].strip()
                entries.append({
                    "key": key,
                    "type": entry_type,
                    "raw": raw_entry
                })
            i = j
        else:
            i = at_pos + 1
    return entries

asli_entries = parse_bibtex_robust(asli_path.read_text(encoding="utf-8"))
tambahan_entries = parse_bibtex_robust(tambahan_path.read_text(encoding="utf-8"))

asli_dict = {e["key"]: e for e in asli_entries}
tambahan_dict = {e["key"]: e for e in tambahan_entries}

# Check duplicates
common_keys = [e["key"] for e in tambahan_entries if e["key"] in asli_dict]
new_keys = [e["key"] for e in tambahan_entries if e["key"] not in asli_dict]
asli_only_keys = [e["key"] for e in asli_entries if e["key"] not in tambahan_dict]

print(f"Total Asli entries: {len(asli_entries)}")
print(f"Total Tambahan entries: {len(tambahan_entries)}")
print(f"Duplicate entries found: {len(common_keys)}")
print(f"New entries from Tambahan: {len(new_keys)}")
print(f"Entries unique to Asli: {len(asli_only_keys)}")

# For merging:
# Follow Asli order first. For common keys, use the Tambahan version (richer with DOI / pages / volume).
merged_entries = []
seen_keys = set()

for e in asli_entries:
    key = e["key"]
    if key in tambahan_dict:
        # Use richer version from Tambahan
        raw = tambahan_dict[key]["raw"]
        merged_entries.append((key, raw, "Updated from Tambahan (richer metadata)"))
    else:
        merged_entries.append((key, e["raw"], "Retained from Asli"))
    seen_keys.add(key)

# Now append truly new entries from Tambahan
for e in tambahan_entries:
    key = e["key"]
    if key not in seen_keys:
        raw = e["raw"]
        # Clean [suspicious link removed] if present
        if "[suspicious link removed]" in raw:
            raw = raw.replace("[suspicious link removed]", "https://doi.org/10.1109/CVPR.2009.5206848")
        merged_entries.append((key, raw, "Added as new from Tambahan"))
        seen_keys.add(key)

print(f"Total merged unique entries: {len(merged_entries)}")

# Write to fr.txt
with open(output_path, "w", encoding="utf-8") as f:
    for idx, (key, raw, source) in enumerate(merged_entries):
        f.write(raw + "\n\n")

print(f"Successfully generated: {output_path}")
