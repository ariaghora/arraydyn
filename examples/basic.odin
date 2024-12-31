package main

import ar "../arraydyn"
import "core:fmt"

main :: proc() {
	a := ar.new_with_init([]f32{1, 2, 3, 4}, {2, 2})
	defer ar.tensor_release(a)

	b := ar.new_with_init([]f32{1, 2, 3, 4}, {2, 2})
	defer ar.tensor_release(b)

	res_add := ar.add(a, b)
	defer ar.tensor_release(res_add)
	ar.print(res_add) // Print tensor

	res_mul := ar.mul(a, b)
	defer ar.tensor_release(res_mul)
	ar.print(res_mul)

	arr_3_1 := ar.new_with_init([]f32{1, 2, 3}, {3, 1})
	defer ar.tensor_release(arr_3_1)
	arr_1_3 := ar.new_with_init([]f32{1, 2, 3}, {1, 3})
	defer ar.tensor_release(arr_1_3)
	res_bcast := ar.add(arr_3_1, arr_1_3)
	defer ar.tensor_release(res_bcast)
	ar.print(res_bcast)
}
