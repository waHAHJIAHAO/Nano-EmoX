def string_to_list(s):
    """Convert a string representation of a list to an actual list."""
    if isinstance(s, list):
        return s
    if not isinstance(s, str):
        return [s]
    s = s.strip()
    if s.startswith('[') and s.endswith(']'):
        s = s[1:-1]
    if len(s) == 0:
        return []
    items = [item.strip().strip("'\"") for item in s.split(',')]
    return [item for item in items if item]


def func_discrte_label_distribution(labels):
    print(f'sample number: {len(labels)}')
    print(f'label number: {len(set(labels))}')
    label2count = {}
    for label in labels:
        if label not in label2count:
            label2count[label] = 0
        label2count[label] += 1
    print('label distribution')
    for label in sorted(label2count):
        print(label, ':', label2count[label])
