import re
import sys

def audit_file(filepath):
    print(f"Auditing GFM KaTeX compliance for: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    issues = []
    in_code_block = False
    in_math_block = False

    for idx, raw_line in enumerate(lines):
        line_num = idx + 1
        line = raw_line.rstrip('\r\n')

        # Check code fence
        if line.strip().startswith('```'):
            if line.strip().startswith('```math'):
                in_math_block = True
                if len(line) - len(line.lstrip()) > 0:
                    issues.append((line_num, "Indented ```math block (must start at Column 0)"))
            elif in_math_block and line.strip() == '```':
                in_math_block = False
            elif in_code_block and line.strip() == '```':
                in_code_block = False
            else:
                in_code_block = True
            continue

        if in_code_block and not in_math_block:
            continue

        # 1. Backtick math
        if '$`' in line or '`$' in line:
            issues.append((line_num, "Backtick math found"))

        # 2. Markdown bold/italics containing $
        if re.search(r'\*\*[^$\n]*\$[^$\n]*\*\*', line) or re.search(r'\*[^\s$\n]*\$[^\s$\n]*\*', line):
            issues.append((line_num, "Markdown bold/italics wrapping math delimiter"))

        # 3. \text{--} in math
        if r'\text{--}' in line:
            issues.append((line_num, r"\text{--} ligature in math"))

        # 5. Raw | inside table math cells
        if line.strip().startswith('|') and '$' in line:
            cells = line.split('|')[1:-1]
            for c in cells:
                if c.count('$') % 2 != 0:
                    issues.append((line_num, "Raw pipe '|' inside table math cell (use \\mid or \\lvert...\\rvert)"))
                    break

        # 7. \hline in markdown table
        if r'\hline' in line:
            issues.append((line_num, r"\hline in markdown table"))

        # 11. \operatorname
        if r'\operatorname' in line:
            issues.append((line_num, r"\operatorname is disallowed in GitHub KaTeX (use \mathrm)"))

        # 15. \left\{ or \right\} in math
        if r'\left\{' in line or r'\right\}' in line:
            issues.append((line_num, r"\left\{ or \right\} in math (use \left\lbrace and \right\rbrace)"))

        # 16. Brace-preceded font macro subscripts
        if re.search(r'\\(mathcal|mathbb|mathbf|mathfrak)\{[A-Z]\}_[a-zA-Z0-9]', line):
            issues.append((line_num, "Brace-preceded font macro subscript (e.g. \\mathcal{H}_L -> \\mathcal H_L)"))

        # 17. Math inside markdown link
        if re.search(r'\[[^\]]*\$[^\]]*\]\(.*?\)', line):
            issues.append((line_num, "Math inside markdown link anchor text"))

        # 20. Unescaped & inside \text
        if re.search(r'\\text\{[^}]*?[^\\]&.*?\}', line):
            issues.append((line_num, "Unescaped literal '&' inside \\text{...}"))

        # 21. Raw < followed by letter inside math (CommonMark HTML tag collision)
        if in_math_block and re.search(r'<[a-zA-Z]', line):
            issues.append((line_num, "Raw '<' followed by letter in display math (triggers HTML tag collision, use \\lt)"))
        elif '$' in line:
            math_spans = re.findall(r'\$(.*?)\$', line)
            for span in math_spans:
                if re.search(r'<[a-zA-Z]', span):
                    issues.append((line_num, "Raw '<' followed by letter in inline math (triggers HTML tag collision, use \\lt)"))

    if not issues:
        print("PASS: 0 KaTeX/GFM compliance issues detected!")
        return 0
    else:
        print(f"FAILED: {len(issues)} compliance issues detected:")
        for l_num, desc in issues:
            print(f"  Line {l_num}: {desc}")
        return 1

if __name__ == '__main__':
    target = sys.argv[1] if len(sys.argv) > 1 else 'docs/preprint/README.md'
    sys.exit(audit_file(target))
