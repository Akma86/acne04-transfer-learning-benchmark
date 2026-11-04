import glob
import re
import os
import fitz  # PyMuPDF

def clean(text):
    return re.sub(r"\s+", " ", text).strip()

# 1. Inspect all PDFs in docs/references
pdf_files = sorted(glob.glob("docs/references/*.pdf"))
pdf_info = []

for p in pdf_files:
    fname = os.path.basename(p)
    doc = fitz.open(p)
    first_page_text = clean(doc[0].get_text()[:1500])
    doc.close()
    pdf_info.append({
        "filename": fname,
        "first_page": first_page_text
    })

# 2. Get all \cite{...} in docs/paper/*.tex
tex_files = glob.glob("docs/paper/*.tex")
cites_used = {}
for tf in tex_files:
    tname = os.path.basename(tf)
    with open(tf, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
        matches = re.finditer(r"\\cite\{([^}]+)\}", content)
        for m in matches:
            line_no = content[:m.start()].count("\n") + 1
            for k in m.group(1).split(","):
                k = k.strip()
                if k not in cites_used:
                    cites_used[k] = []
                cites_used[k].append(f"{tname}:{line_no}")

# 3. Read References.bib
with open("docs/paper/References.bib", "r", encoding="utf-8", errors="ignore") as f:
    bib_content = f.read()

entries = re.findall(r"@(\w+)\s*\{\s*([^,]+),([\s\S]*?)(?=@\w+\s*\{|\Z)", bib_content)
bib_dict = {}
for entry_type, key, body in entries:
    key = key.strip()
    title_match = re.search(r"title\s*=\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", body, re.IGNORECASE) or re.search(r"title\s*=\s*[\"{](.*?)[\"}],?", body, re.IGNORECASE)
    author_match = re.search(r"author\s*=\s*[\"{](.*?)[\"}],?", body, re.IGNORECASE)
    year_match = re.search(r"year\s*=\s*[\"{]?(\d{4})[\"}],?", body, re.IGNORECASE)
    title = clean(title_match.group(1).replace("{", "").replace("}", "")) if title_match else ""
    author = clean(author_match.group(1).replace("{", "").replace("}", "")) if author_match else ""
    year = year_match.group(1) if year_match else ""
    bib_dict[key] = {"type": entry_type, "title": title, "author": author, "year": year}

# 4. Detailed Matching
results = []
for k in sorted(cites_used.keys()):
    b = bib_dict.get(k, {})
    title = b.get("title", "")
    author = b.get("author", "")
    year = b.get("year", "")
    locations = cites_used[k]
    
    # Try finding matching PDF
    matched_pdf = None
    match_reason = ""
    
    # Normalize for comparison
    def norm(t):
        return re.sub(r"[^a-zA-Z0-9]", "", t).lower()
        
    norm_title = norm(title)
    
    # Check by filename first
    for p in pdf_info:
        fn_norm = norm(p["filename"].replace(".pdf", ""))
        if len(norm_title) > 8 and (norm_title in fn_norm or fn_norm in norm_title):
            matched_pdf = p["filename"]
            match_reason = "Direct filename match"
            break
            
    # Check by author or first page text if not matched
    if not matched_pdf:
        # Check author last name in first page
        author_last = author.split(",")[0].split()[0] if author else ""
        for p in pdf_info:
            # check if title words or author appear in first page
            title_words = [w.lower() for w in re.findall(r"\w+", title) if len(w) > 3]
            match_count = sum(1 for w in title_words if w in p["first_page"].lower())
            if len(title_words) >= 3 and match_count >= len(title_words) * 0.7:
                matched_pdf = p["filename"]
                match_reason = "First page content match (title words)"
                break
            elif author_last and len(author_last) > 3 and author_last.lower() in p["first_page"].lower():
                if len(title_words) >= 2 and sum(1 for w in title_words if w in p["first_page"].lower()) >= 2:
                    matched_pdf = p["filename"]
                    match_reason = "Author + partial title match in PDF"
                    break

    results.append({
        "key": k,
        "author": author,
        "year": year,
        "title": title,
        "locations": locations,
        "matched_pdf": matched_pdf,
        "match_reason": match_reason
    })

# Write comprehensive report to file
with open("scripts/reference_audit_report.txt", "w", encoding="utf-8") as out:
    out.write(f"TOTAL ACTIVE CITES IN PAPER: {len(cites_used)}\n")
    out.write(f"TOTAL PDFS IN docs/references: {len(pdf_info)}\n")
    out.write("="*90 + "\n\n")
    
    matched_count = sum(1 for r in results if r["matched_pdf"])
    out.write(f"STATUS SUMMARY: {matched_count} / {len(results)} CITATIONS HAVE PDF FILES IN docs/references\n\n")
    
    out.write("--- 1. CITATIONS DENGAN FILE PDF (SUDAH ADA DI FOLDER) ---\n")
    for r in results:
        if r["matched_pdf"]:
            out.write(f"[ADA] {r['key']}\n")
            out.write(f"      Penulis : {r['author']}\n")
            out.write(f"      Judul   : {r['title']}\n")
            out.write(f"      File PDF: {r['matched_pdf']}\n")
            out.write(f"      Lokasi  : {', '.join(r['locations'])}\n\n")
            
    out.write("\n" + "="*90 + "\n")
    out.write("--- 2. CITATIONS YANG BELUM ADA FILE PDF-NYA DI docs/references ---\n")
    missing_count = 0
    for r in results:
        if not r["matched_pdf"]:
            missing_count += 1
            out.write(f"[TIDAK ADA PDF] {r['key']}\n")
            out.write(f"      Penulis : {r['author']}\n")
            out.write(f"      Tahun   : {r['year']}\n")
            out.write(f"      Judul   : {r['title']}\n")
            out.write(f"      Lokasi  : {', '.join(r['locations'])}\n\n")
            
    out.write("\n" + "="*90 + "\n")
    out.write("--- 3. FILE PDF DI docs/references YANG BELUM DISITASI DI PAPER ---\n")
    used_pdfs = set(r["matched_pdf"] for r in results if r["matched_pdf"])
    for p in pdf_info:
        if p["filename"] not in used_pdfs:
            out.write(f"- {p['filename']}\n")

print("Audit report written to scripts/reference_audit_report.txt")
