package arraydyn


import "core:fmt"
import "core:math/rand"
import "core:slice"

// multinomial_a samples indices according to the probabilities given in arr.
//
// Parameters:
//   arr: 2D tensor of shape [batch_size, num_categories] containing probabilities
//        that sum to 1 along the last dimension
//   num_samples: Number of indices to sample for each batch
//   replacement: If true, sample with replacement allowing the same index multiple times
//               If false, sample without replacement (unique indices)
//
// Returns:
//   A tensor of shape [batch_size, num_samples] containing sampled indices
//
// Example:
//   probs = [[0.3, 0.7], [0.6, 0.4]]  // 2 batches, 2 categories each
//   samples = multinomial(probs, 3)    // Sample 3 indices per batch
//   samples = [[1, 1, 0], [0, 0, 1]]   // Possible output based on probabilities
multinomial_a :: proc(
	arr: ^Array_Dyn($T),
	num_samples: uint,
	replacement := false,
) -> ^Array_Dyn(T) {
	if len(arr.shape) != 2 {
		panic("Input must be 2D tensor with shape [batch_size, num_categories]")
	}

	batch_size := uint(1)
	num_categories := arr.shape[0]
	if len(arr.shape) == 2 {
		batch_size = arr.shape[0]
		num_categories = arr.shape[1]
	}

	// Output shape and initialization
	out_shape := make([]uint, len(arr.shape))
	defer delete(out_shape)
	if len(arr.shape) == 1 {
		out_shape[0] = num_samples
	} else {
		out_shape[0] = batch_size
		out_shape[1] = num_samples
	}
	result := _array_alloc(T, out_shape)

	for b: uint = 0; b < batch_size; b += 1 {
		// Normalize probabilities
		sum_p: T = 0
		for j: uint = 0; j < num_categories; j += 1 {
			sum_p += array_get(arr, b if len(arr.shape) == 2 else 0, j)
		}

		available := make([]bool, num_categories)
		defer delete(available)
		slice.fill(available, true)

		for i: uint = 0; i < num_samples; i += 1 {
			r := T(rand.float64())
			cumsum: T = 0

			for j: uint = 0; j < num_categories; j += 1 {
				if replacement || available[j] {
					prob := array_get(arr, b if len(arr.shape) == 2 else 0, j) / sum_p
					cumsum += prob
					if r < cumsum {
						// Fixed indexing
						if len(arr.shape) == 1 {
							result.data[i] = T(j)
						} else {
							result.data[b * num_samples + i] = T(j)
						}
						if !replacement {
							available[j] = false
						}
						break
					}
				}
			}
		}
	}

	return result
}
