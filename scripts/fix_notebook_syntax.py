import json

nb_path = "notebooks/03_phase2_bihemispheric_scaling_arc2_colab.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

found_issues = []
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        # Check if python compile throws error
        try:
            compile(source, f"cell_{i}.py", "exec")
        except SyntaxError as e:
            found_issues.append((i, str(e), e.lineno, e.text))
            print(f"Cell {i} SyntaxError: {e} at line {e.lineno}: {e.text}")

print(f"Total syntax errors found: {len(found_issues)}")
