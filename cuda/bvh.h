#ifndef BVH_H
#define BVH_H

#include "hitable.h"
#include "aabb.h"

// Flat BVH Node for GPU traversal (no pointers, just array indices)
struct BVHFlatNode {
    aabb bounds;
    int left;         // Index of left child, or primitive index if leaf
    int right;        // Index of right child, or -1 if leaf
    int is_leaf;      // Is this node a leaf (use int instead of bool for alignment)
};

// GPU BVH class - uses flat array representation for stack-based traversal
class bvh : public hitable {
public:
    BVHFlatNode* nodes;      // Array of BVH nodes (device memory)
    hitable** primitives;    // Array of pointers to primitives (device memory)
    int num_nodes;           // Total number of nodes
    int root_idx;            // Index of root node (usually 0)

    __device__ bvh() : nodes(nullptr), primitives(nullptr), num_nodes(0), root_idx(0) {}

    __device__ bvh(BVHFlatNode* n, hitable** p, int count, int root)
        : nodes(n), primitives(p), num_nodes(count), root_idx(root) {}

    __device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const {
        if (nodes == nullptr || num_nodes == 0) {
            return false;
        }

        // Iterative traversal with explicit stack (no recursion on GPU)
        // Reduced stack size: log2(975) ≈ 10, so 32 is plenty
        int stack[32];
        int stack_ptr = 0;
        stack[stack_ptr++] = root_idx;

        bool hit_anything = false;
        float closest_so_far = tmax;

        // Safety counter to prevent infinite loops
        int max_iterations = num_nodes * 2;
        int iterations = 0;

        while (stack_ptr > 0 && iterations < max_iterations) {
            iterations++;
            int node_idx = stack[--stack_ptr];

            // Bounds check
            if (node_idx < 0 || node_idx >= num_nodes) {
                continue;
            }

            const BVHFlatNode& node = nodes[node_idx];

            // Test ray against node's bounding box
            if (!node.bounds.hit(r, tmin, closest_so_far)) {
                continue;  // Skip this subtree
            }

            if (node.is_leaf) {
                // Leaf node - test the primitive
                int prim_idx = node.left;
                if (prim_idx >= 0 && primitives[prim_idx]->hit(r, tmin, closest_so_far, rec)) {
                    hit_anything = true;
                    closest_so_far = rec.t;
                }
            } else {
                // Internal node - push children onto stack
                if (stack_ptr < 30) {  // Leave room for 2 more entries
                    if (node.right >= 0 && node.right < num_nodes) stack[stack_ptr++] = node.right;
                    if (node.left >= 0 && node.left < num_nodes) stack[stack_ptr++] = node.left;
                }
            }
        }

        return hit_anything;
    }
};

#endif