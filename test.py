max_item_id=24915
entity_counts={}
import numpy as np
file_name = "kg_final.txt"
kg_np = np.loadtxt(file_name, dtype=np.int32)
for head, relation, tail in kg_np:
    # if tail >= max_item_id and head < max_item_id:
    if tail >= max_item_id:
        if tail not in entity_counts:
            entity_counts[tail] = 0
        entity_counts[tail] += 1
    # if head >= max_item_id and tail < max_item_id:
    if head >= max_item_id:
        if head not in entity_counts:
            entity_counts[head] = 0
        entity_counts[head] += 1
entity_counts= sorted(entity_counts.items(), key=lambda x: x[1], reverse=False)
print entity_counts