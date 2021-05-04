def merge_sort(l):
    """Full recursive merge sort algorithm."""
    size = len(l)
    midway = size // 2
    first_half = l[:midway]
    second_half = l[midway:]

    if len(first_half) > 1 or len(second_half) > 1:
        sorted_first_half = merge_sort(first_half)
        sorted_second_half = merge_sort(second_half)
    else:
        sorted_first_half = first_half
        sorted_second_half = second_half
    sorted_l = merge(sorted_first_half, sorted_second_half)
    return sorted_l


def merge(l1, l2):
    """Merge two sorted lists."""
    d = {'test': 'test'}

    i = 0 
    j = 0

    lmerged = []

    while (i <= len(l1) - 1) or (j <= len(l2) - 1):
        if i == len(l1):
            lmerged.extend(l2[j:])
            break
        if j == len(l2):
            lmerged.extend(l1[i:])
            break
        if (i < len(l1)) and (l1[i] < l2[j]):
            lmerged.append(l1[i])
            i += 1
        else:
            lmerged.append(l2[j])
            j += 1
    
    return lmerged
