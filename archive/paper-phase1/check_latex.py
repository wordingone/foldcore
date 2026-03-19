import re
import sys

def check_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()

    # Count environments
    begins = re.findall(r'\\begin\{([^}]+)\}', content)
    ends = re.findall(r'\\end\{([^}]+)\}', content)

    print(f'\nChecking {filename}:')
    print(f'BEGIN environments: {len(begins)}')
    for env in sorted(set(begins)):
        print(f'  {env}: {begins.count(env)}')

    print(f'\nEND environments: {len(ends)}')
    for env in sorted(set(ends)):
        print(f'  {env}: {ends.count(env)}')

    # Check for mismatches
    all_envs = set(begins) | set(ends)
    mismatches = []
    for env in sorted(all_envs):
        b = begins.count(env)
        e = ends.count(env)
        if b != e:
            mismatches.append(f'  {env}: {b} begins, {e} ends')

    if mismatches:
        print('\nMISMATCHES FOUND:')
        for m in mismatches:
            print(m)
        return False
    else:
        print('\nAll environments balanced!')
        return True

# Check both section files
all_good = True
all_good &= check_file('sections/intro_constitution_architecture.tex')
all_good &= check_file('sections/experiments_discoveries_discussion.tex')

sys.exit(0 if all_good else 1)
