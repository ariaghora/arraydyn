package arraydyn

import "base:intrinsics"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"
import "core:slice"
import "core:strings"

// This structure implements a high level multidimensional array container using a
// linear array of data internally. The array is parametrized for any data type
// and supports strided access and broadcasting. The internal representation uses
// a C-contiguous storage layout (row-major order) with all the data stored in a
// single slice. A stride array is used to map N-dimensional coordinates to linear
// indices in the data array. The contiguous flag indicates if the array is stored
// in memory without gaps.
Array_Dyn :: struct($T: typeid) where intrinsics.type_is_numeric(T) {
	data:       []T,
	shape:      []uint,
	strides:    []uint,
	contiguous: bool,
	own_data:   bool,
}

// Tensor represents a multidimensional array with automatic differentiation capabilities.
// It wraps an Array_Dyn for data storage and adds gradient computation functionality.
// The struct tracks computational dependencies through a dynamic array of dependencies
// and supports automatic gradient propagation via a backward function.
Tensor :: struct($T: typeid) where intrinsics.type_is_numeric(T) {
	using arrdata:    ^Array_Dyn(T),
	grad:             ^Array_Dyn(T),
	deps:             [dynamic]^Tensor(T),
	visited:          bool,
	requires_grad:    bool,
	backward_fn:      proc(_: ^Tensor(T), _: ^Array_Dyn(T)),
	backward_fn_name: string,
	ref_count:        uint,
}

// Compute total size of an array by multiplying dimensions in shape
_shape_to_size :: #force_inline proc(shape: []uint) -> uint {
	size: uint = 1
	for s in shape {size *= s}
	return size
}

// Create a new n-dimensional array with the given shape. For each dimension i
// in the array shape[i] represents the size of that dimension. After allocation
// the array elements are left uninitialized.
_array_alloc :: proc($T: typeid, shape: []uint) -> (res: ^Array_Dyn(T)) {
	res = new(Array_Dyn(T))
	size := _shape_to_size(shape)
	res.data = make([]T, size)
	res.shape = make([]uint, len(shape))
	res.strides = make([]uint, len(shape))
	res.contiguous = true
	res.own_data = true

	// initialize shape and strides
	copy(res.shape, shape)
	stride: uint = 1
	for i := len(shape) - 1; i >= 0; i -= 1 {
		res.strides[i] = stride
		stride *= shape[i]
	}
	return res
}

// Get array data respecting strides. If shape and strides match the original
// array and data is contiguous, return a simple clone. Otherwise rearrange
// the data following target shape and strides to handle non-contiguous cases.
// Under the hood this function converts non-contiguous array storage into
// contiguous one.
_get_strided_data :: proc(arr: ^Array_Dyn($T), shape: []uint = nil, strides: []uint = nil) -> []T {
	target_strides := strides
	target_shape := shape
	if (strides == nil) {
		target_strides = arr.strides
	}
	if (shape == nil) {
		target_shape = arr.shape
	}

	if arr.contiguous &&
	   (slice.equal(arr.shape, target_shape) || shape == nil) &&
	   (slice.equal(arr.strides, target_strides) || strides == nil) {
		return slice.clone(arr.data)
	}

	// return strided
	size := _shape_to_size(target_shape)
	data := make([]T, size)
	i: uint
	for i in 0 ..< size {
		i_strided := _compute_strided_index(target_shape, target_strides, i)
		data[i] = arr.data[i_strided]
	}
	return data
}

// Creates a copy of an array with either shared or independent data storage.
// If deep=true, creates a completely independent copy with its own data allocation.
// If deep=false, creates a view that shares the underlying data with the source array.
// In both cases, shape and strides are independently copied.
clone :: proc(arr: ^Array_Dyn($T), deep: bool = false) -> (res: ^Array_Dyn(T)) {
	res = new(Array_Dyn(T))

	if deep {
		res.data = make([]T, len(arr.data))
		copy(res.data, arr.data)
	} else {
		res.data = arr.data
	}

	res.shape = make([]uint, len(arr.shape))
	res.strides = make([]uint, len(arr.strides))
	res.contiguous = arr.contiguous

	copy(res.shape, arr.shape)
	copy(res.strides, arr.strides)

	return res
}

// Returns the total number of elements in the array by multiplying all dimensions
// together. Takes into account that some dimensions may be empty (0) in which
// case the total size will also be zero. This function does not care about strides
// or how data is laid out in memory, it is just about the logical size.
data_len :: proc(arr: ^Array_Dyn($T)) -> uint {
	return _shape_to_size(arr.shape)
}

// Create a new array filled with ones. Arrays are initialized for all data types by casting
// 1 to the target type, so for example this works with floating point data types, integers
// and even complex data types like bool or void types.
_ones :: proc($T: typeid, shape: []uint) -> (res: ^Array_Dyn(T)) {
	res = _array_alloc(T, shape)
	for _, i in res.data {res.data[i] = T(1)}
	return res
}

ones :: proc($T: typeid, shape: []uint) -> (res: ^Tensor(T)) {
	return _tensor_from_array(_ones(T, shape))
}

// Create an array with normally-distributed random values
_randn :: proc($T: typeid, shape: []uint, mean, stddev: T) -> (res: ^Array_Dyn(T)) {
	res = _array_alloc(T, shape)
	for _, i in res.data {
		// Box-Muller transform to generate normal distribution
		u1 := rand.float64()
		u2 := rand.float64()
		z := math.sqrt(-2 * math.ln(u1)) * math.cos(2 * math.PI * u2)
		res.data[i] = T(z)
	}
	return res
}
randn :: proc($T: typeid, shape: []uint, mean, stddev: T) -> (res: ^Tensor(T)) {
	return _tensor_from_array(_randn(T, shape, mean, stddev))
}

// Create a new array filled with zeros. Arrays are initialized for all data types by casting
// 0 to the target type, so for example this works with floating point data types, integers
// and even complex data types like bool or void types.
_zeros :: proc($T: typeid, shape: []uint) -> (res: ^Array_Dyn(T)) {
	res = _array_alloc(T, shape)
	for _, i in res.data {res.data[i] = T(0)}
	return res
}
zeros :: proc($T: typeid, shape: []uint) -> (res: ^Tensor(T)) {
	return _tensor_from_array(_zeros(T, shape))
}

// Create a new array with given data and shape. This function performs a copy
// of the input data, so the original array is not referenced in the new one.
_new_with_init :: proc(init: []$T, shape: []uint) -> (res: ^Array_Dyn(T)) {
	res = _array_alloc(T, shape)
	if len(res.data) != len(init) {
		panic("Input data length must match array size computed from shape")
	}

	copy(res.data, init)
	return res
}

new_with_init :: proc(init: []$T, shape: []uint) -> (res: ^Tensor(T)) {
	res = new(Tensor(T))
	res.ref_count = 1
	res.arrdata = _new_with_init(init, shape)
	return res
}

_tensor_from_array :: proc(arr: ^Array_Dyn($T)) -> (res: ^Tensor(T)) {
	res = new(Tensor(T))
	res.ref_count = 1
	res.arrdata = arr
	return res
}


// Pretty print array with numpy-like formatting
print :: proc {
	print_arr,
	print_tensor,
}

@(private = "file")
print_tensor :: proc(t: ^Tensor($T)) {
	print_arr(t.arrdata, prefix = "Tensor", backward_fn_name = t.backward_fn_name)
}

@(private = "file")
print_arr :: proc(arr: ^Array_Dyn($T), prefix: string = "Array", backward_fn_name: string = "") {
	if len(arr.shape) == 0 {
		fmt.printf("%s()", prefix)
		return
	}

	builder := strings.builder_make()
	defer strings.builder_destroy(&builder)

	backward_fn_str :=
		len(backward_fn_name) > 0 ? fmt.tprintf(", backward_fn=%s", backward_fn_name) : ""
	strings.write_string(
		&builder,
		fmt.tprintf("%s(type=%v, shape=%v%s)\n", prefix, typeid_of(T), arr.shape, backward_fn_str),
	)

	indices := make([]uint, len(arr.shape))
	defer delete(indices)
	print_recursive(arr, arr.shape, arr.strides, 0, indices, 1, &builder)
	strings.write_string(&builder, "\n")

	fmt.println(strings.to_string(builder))
}

@(private = "file")
print_recursive :: proc(
	arr: ^Array_Dyn($T),
	shape: []uint,
	strides: []uint,
	depth: int,
	indices: []uint,
	indent: int,
	builder: ^strings.Builder,
) {
	if depth == len(shape) - 1 {
		// For the innermost dimension
		strings.write_byte(builder, '[')

		for i in 0 ..< shape[depth] {
			indices[depth] = i
			index := compute_linear_index(indices, strides)

			if i > 0 {
				strings.write_byte(builder, ' ')
			}

			value := arr.data[index]
			switch type_info_of(type_of(value)) {
			case type_info_of(f32):
				strings.write_string(builder, fmt.tprintf("%.6f", value))
			case type_info_of(f64):
				strings.write_string(builder, fmt.tprintf("%.6f", value))
			case:
				strings.write_string(builder, fmt.tprintf("% 6v", value))
			}
		}
		strings.write_byte(builder, ']')
	} else {
		strings.write_byte(builder, '[')

		for i in 0 ..< shape[depth] {
			if i > 0 {
				strings.write_string(builder, "\n")
				for j in 0 ..< depth + 1 {
					strings.write_string(builder, " ")
				}
			}

			indices[depth] = i
			print_recursive(arr, shape, strides, depth + 1, indices, indent + 1, builder)

			if i < shape[depth] - 1 {
				strings.write_byte(builder, ',')
			}
		}
		strings.write_string(builder, "]")
	}
}

@(private = "package")
compute_linear_index :: proc(indices: []uint, strides: []uint) -> uint {
	index: uint = 0
	for i in 0 ..< len(indices) {
		index += indices[i] * strides[i]
	}
	return index
}

array_free :: proc {
	array_free_one,
	array_free_many,
}

@(private = "file")
array_free_one :: proc(arr: ^Array_Dyn($T)) {
	if arr.own_data {
		delete(arr.data)
	}
	delete(arr.shape)
	delete(arr.strides)
	free(arr)
}

@(private = "file")
array_free_many :: proc(arr: ^Array_Dyn($T), rest: ..^Array_Dyn(T)) {
	array_free_one(arr)
	for r in rest {
		array_free_one(r)
	}
}

tensor_release :: proc {
	tensor_release_one,
	tensor_release_many,
}

tensor_release_one :: proc(t: ^Tensor($T)) {
	if t == nil do return

	t.ref_count -= 1
	if t.ref_count == 0 {
		// Actually free resources
		array_free(t.arrdata)
		if t.requires_grad {
			array_free(t.grad)
		}
		// Decrease ref count for dependencies
		for dep in t.deps {
			tensor_release(dep)
		}
		delete(t.deps)
		free(t)
	}
}

tensor_release_many :: proc(t: ^Tensor($T), rest: ..^Tensor(T)) {
	tensor_release_one(t)
	for r in rest {
		tensor_release_one(r)
	}
}

array_get :: proc {
	_array_get_1d,
	_array_get_2d,
	_array_get_3d,
	_array_get_4d,
	_array_get_nd,
}

_array_get_1d :: #force_inline proc(arr: ^Array_Dyn($T), i0: uint) -> T {
	s0 := arr.strides[0]
	return arr.data[i0 * s0]
}

_array_get_2d :: #force_inline proc(arr: ^Array_Dyn($T), row, col: uint) -> T {
	s0, s1 := arr.strides[0], arr.strides[1]
	return arr.data[row * s0 + col * s1]
}

_array_get_3d :: #force_inline proc(arr: ^Array_Dyn($T), i0, i1, i2: uint) -> T {
	s0, s1, s2 := arr.strides[0], arr.strides[1], arr.strides[2]
	return arr.data[i0 * s0 + i1 * s1 + i2 * s2]
}

_array_get_4d :: #force_inline proc(arr: ^Array_Dyn($T), i0, i1, i2, i3: uint) -> T {
	s0, s1, s2, s3 := arr.strides[0], arr.strides[1], arr.strides[2], arr.strides[3]
	return arr.data[i0 * s0 + i1 * s1 + i2 * s2 + i3 * s3]
}

_array_get_nd :: #force_inline proc(arr: ^Array_Dyn($T), coord: []uint) -> T {
	index: uint = 0
	for i in 0 ..< len(coord) {
		index += coord[i] * arr.strides[i]
	}
	return arr.data[index]
}


_compute_strided_index :: #force_inline proc(shape, strides: []uint, idx: uint) -> uint {
	#no_bounds_check {
		switch len(shape) {
		case 1:
			return idx * strides[0]
		case 2:
			s0, s1 := strides[0], strides[1]
			d1 := shape[1]
			return (idx / d1) * s0 + (idx % d1) * s1
		case 4:
			s0, s1, s2, s3 := strides[0], strides[1], strides[2], strides[3]
			d1, d2, d3 := shape[1], shape[2], shape[3]
			i3 := idx % d3
			tmp := idx / d3
			i2 := tmp % d2
			tmp /= d2
			i1 := tmp % d1
			i0 := tmp / d1
			return i0 * s0 + i1 * s1 + i2 * s2 + i3 * s3
		case:
			// For N-dim, precompute products to avoid repeated divisions
			offset: uint = 0
			remaining := idx
			dim_product := uint(1)
			for i := len(shape) - 1; i >= 0; i -= 1 {
				coord := (remaining / dim_product) % shape[i]
				offset += coord * strides[i]
				dim_product *= shape[i]
			}
			return offset
		}
	}
}

reshape :: proc {
	reshape_a,
	reshape_t,
}

reshape_a :: proc(arr: ^Array_Dyn($T), new_shape: []uint) -> (res: ^Array_Dyn(T)) {
	// Check if total size matches
	old_size := _shape_to_size(arr.shape)
	new_size := _shape_to_size(new_shape)
	if old_size != new_size {
		panic(fmt.tprintf("Cannot reshape array of size %v to shape %v", old_size, new_shape))
	}

	res = clone(arr, deep = false)
	delete(res.shape)
	delete(res.strides)

	// Create new shape and strides
	res.shape = make([]uint, len(new_shape))
	res.strides = make([]uint, len(new_shape))
	copy(res.shape, new_shape)

	// Calculate new strides
	stride: uint = 1
	for i := len(new_shape) - 1; i >= 0; i -= 1 {
		res.strides[i] = stride
		stride *= new_shape[i]
	}

	return res
}

reshape_t :: proc(t: ^Tensor($T), new_shape: []uint) -> ^Tensor(T) {
	return autograd_make_op(
		[]^Tensor(T){t},
		new_arrdata = reshape_a(t.arrdata, new_shape),
		backward_fn_name = "reshape_backward",
		backward_fn = proc(tensor: ^Tensor(T), upstream_grad: ^Array_Dyn(T)) {
			input := tensor.deps[0]
			if input.requires_grad {
				old_grad := input.grad
				reshaped_grad := reshape_a(upstream_grad, input.shape)
				input.grad = add(old_grad, reshaped_grad)
				array_free(old_grad, reshaped_grad)
			}
		},
	)
}

Range :: struct {
	start, end: uint,
}

slice :: proc {
	slice_a,
	slice_t,
}

slice_a :: proc(arr: ^Array_Dyn($T), ranges: ..Range) -> ^Array_Dyn(T) {
	// Validate inputs
	if len(ranges) > len(arr.shape) {
		panic("Too many ranges specified")
	}

	// For unspecified dimensions, use full range
	new_shape := make([]uint, len(arr.shape))
	new_strides := make([]uint, len(arr.shape))
	offset: uint = 0

	for i := 0; i < len(arr.shape); i += 1 {
		if i < len(ranges) {
			// Validate range
			if ranges[i].end > arr.shape[i] {
				panic("Slice index out of bounds")
			}
			if ranges[i].start > ranges[i].end {
				panic("Invalid slice range: start > end")
			}

			// Calculate new shape and offset
			new_shape[i] = ranges[i].end - ranges[i].start
			new_strides[i] = arr.strides[i]
			offset += ranges[i].start * arr.strides[i]
		} else {
			// Use full range for unspecified dimensions
			new_shape[i] = arr.shape[i]
			new_strides[i] = arr.strides[i]
		}
	}

	// Create view into original data
	result := new(Array_Dyn(T))
	result.data = arr.data[offset:]
	result.shape = new_shape
	result.strides = new_strides
	result.contiguous = false // Slicing generally creates non-contiguous view
	result.own_data = false

	return result
}

slice_t :: proc(t: ^Tensor($T), ranges: ..Range) -> ^Tensor(T) {
	return autograd_make_op(
		[]^Tensor(T){t},
		new_arrdata = slice_a(t.arrdata, ..ranges),
		backward_fn_name = "slice_backward",
		backward_fn = nil,
	)
}


set_requires_grad :: proc(t: ^Tensor($T), val: bool) {
	if val {
		// if val is true and  arr already requires grad, nothing to do
		if t.requires_grad {return}

		// set grads to 0 with shape == arr.shape
		t.grad = _zeros(T, t.shape)
		t.requires_grad = true
	} else {
		// if arr originally requires grad, free it
		if t.requires_grad {
			array_free(t.grad)
		}
		t.requires_grad = false
	}
}
