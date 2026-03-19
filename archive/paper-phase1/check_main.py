import re

with open('paper.tex', 'r', encoding='utf-8') as f:
    content = f.read()

# Check document structure
has_begin_doc = r'\begin{document}' in content
has_end_doc = r'\end{document}' in content

print(f'Document structure:')
print(f'  \\begin{{document}}: {has_begin_doc}')
print(f'  \\end{{document}}: {has_end_doc}')

# Count environments
begins = re.findall(r'\\begin\{([^}]+)\}', content)
ends = re.findall(r'\\end\{([^}]+)\}', content)

print(f'\nMain file environments:')
all_envs = sorted(set(begins) | set(ends))
issues = []
for env in all_envs:
    b = begins.count(env)
    e = ends.count(env)
    status = ' <-- MISMATCH' if b != e else ''
    print(f'  {env}: {b} begins, {e} ends{status}')
    if b != e:
        issues.append(f'{env}: {b} begins but {e} ends')

# Check input files
inputs = re.findall(r'\\input\{([^}]+)\}', content)
print(f'\nInput files:')
for inp in inputs:
    print(f'  - {inp}')

# Summary
if issues or not (has_begin_doc and has_end_doc):
    print('\nISSUES FOUND:')
    if not has_begin_doc:
        print('  - Missing \\begin{document}')
    if not has_end_doc:
        print('  - Missing \\end{document}')
    for issue in issues:
        print(f'  - {issue}')
else:
    print('\nNo structural issues found in main file!')
