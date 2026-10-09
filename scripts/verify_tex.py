import re

with open('docs/preprint/bihemispheric_ai_preprint.tex', 'r', encoding='utf-8') as f:
    text = f.read()

# Stack-based environment checker
tokens = re.findall(r'\\(begin|end)\{([a-zA-Z*]+)\}', text)
stack = []
mismatch = False
for action, env in tokens:
    if action == 'begin':
        stack.append(env)
    elif action == 'end':
        if not stack:
            print(f"Error: \\end{{{env}}} with empty stack!")
            mismatch = True
            break
        top = stack.pop()
        if top != env:
            print(f"Mismatch: expected \\end{{{top}}}, found \\end{{{env}}}!")
            mismatch = True
            break

if not mismatch and not stack:
    print(f"All {len(tokens)//2} LaTeX environments nested and matched perfectly!")
elif stack:
    print(f"Unclosed environments remaining on stack: {stack}")

# Accurate line-by-line dollar checker
lines = text.split('\n')
odd_lines = []
for line_idx, line in enumerate(lines):
    # remove escaped \% first
    clean = line.replace('\\%', '')
    # split on real comment %
    comment_idx = clean.find('%')
    if comment_idx != -1:
        clean = clean[:comment_idx]
    dollars = clean.count('$')
    if dollars % 2 != 0:
        odd_lines.append((line_idx + 1, line))

if odd_lines:
    print(f"Odd number of dollars on {len(odd_lines)} lines:")
    for l_num, l_str in odd_lines:
        print(f"  Line {l_num}: {l_str}")
else:
    print("All inline math dollar delimiters paired properly on each line!")
