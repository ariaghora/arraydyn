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
//
// Returns:
//   A tensor of shape [batch_size, num_samples] containing sampled indices
//
// Example:
//   probs = [[0.3, 0.7], [0.6, 0.4]]  // 2 batches, 2 categories each
//   samples = multinomial(probs, 3)    // Sample 3 indices per batch
//   samples = [[1, 1, 0], [0, 0, 1]]   // Possible output based on probabilities
multinomial_a :: proc(arr: ^Array_Dyn($T), num_samples: uint) -> ^Array_Dyn(T) {
	if len(arr.shape) != 2 {
		panic("Input must be 2D tensor with shape [batch_size, num_categories]")
	}

	batch_size := arr.shape[0]
	num_categories := arr.shape[1]
	result := _array_alloc(T, []uint{batch_size, num_samples})

	for b: uint = 0; b < batch_size; b += 1 {
		// First compute total sum for normalization
		total: T = 0
		for j: uint = 0; j < num_categories; j += 1 {
			total += arr.data[b * num_categories + j]
		}

		// Pre-compute normalized cumulative probabilities
		cumsum := make([]T, num_categories)
		defer delete(cumsum)

		cumsum[0] = arr.data[b * num_categories] / total
		for j: uint = 1; j < num_categories; j += 1 {
			cumsum[int(j)] = cumsum[int(j - 1)] + arr.data[b * num_categories + j] / total
		}

		for i: uint = 0; i < num_samples; i += 1 {
			r := T(rand.float64()) // No need to scale r since probs sum to 1

			// Binary search for the interval containing r
			chosen_idx: uint = 0
			for j: uint = 0; j < num_categories; j += 1 {
				if r <= cumsum[int(j)] {
					chosen_idx = j
					break
				}
			}
			result.data[b * num_samples + i] = T(chosen_idx)
		}
	}

	return result
}
