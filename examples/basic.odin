package main

import ar "../arraydyn"
import "core:fmt"

main :: proc() {
	a := ar.new_with_init([]f32{1, 2, 3, 4}, {2, 2})
	defer ar.array_free(a)

	b := ar.new_with_init([]f32{1, 2, 3, 4}, {2, 2})
	defer ar.array_free(b)

	res_add := ar.add(a, b)
	defer ar.array_free(res_add)

	res_mul := ar.mul(a, b)
	defer ar.array_free(res_mul)

	fmt.println(res_add)
	fmt.println(res_mul)
}
