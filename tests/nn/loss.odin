package tests_nn

import ar "../../arraydyn"
import nn "../../arraydyn/nn"
import "core:fmt"
import "core:slice"
import "core:testing"

@(test)
test_ce_perfect_prediction :: proc(t: ^testing.T) {
    // Case where model predicts correct class with 100% confidence
    logits := ar.new_with_init(
        []f64{100.0, 0.0, 0.0,  // High confidence for class 0
              0.0, 100.0, 0.0}, // High confidence for class 1
        {2, 3},
    )
    targets := ar.new_with_init([]f64{0, 1}, {2})  // Correct labels
    defer ar.tensor_release(logits, targets)

    loss := nn.loss_crossentropy_with_logit(logits, targets)
    defer ar.tensor_release(loss)

    // Loss should be very small (not exactly 0 due to numerical stability)
    testing.expect(
        t,
        loss.data[0] < 1e-5,
        fmt.tprintf("Expected very small loss, got %v", loss.data[0]),
    )
}

@(test)
test_ce_wrong_prediction :: proc(t: ^testing.T) {
    // Case where model predicts wrong class with high confidence
    logits := ar.new_with_init(
        []f32{0.0, 100.0, 0.0,  // High confidence for class 1
              0.0, 100.0, 0.0},  // High confidence for class 1
        {2, 3},
    )
    targets := ar.new_with_init([]f32{0, 2}, {2})  // But true classes are 0 and 2
    defer ar.tensor_release(logits, targets)

    loss := nn.loss_crossentropy_with_logit(logits, targets)
    defer ar.tensor_release(loss)

    // Loss should be high
    testing.expect(
        t,
        loss.data[0] > 5.0,
        fmt.tprintf("Expected high loss, got %v", loss.data[0]),
    )
}

@(test)
test_ce_gradient :: proc(t: ^testing.T) {
    logits := ar.new_with_init(
        []f32{2.0, -1.0, 0.5}, // Single example
        {1, 3},
    )
    targets := ar.new_with_init([]f32{0}, {1})  // True class is 0
    defer ar.tensor_release(logits, targets)

    ar.set_requires_grad(logits, true)
    loss := nn.loss_crossentropy_with_logit(logits, targets)
    ar.backward(loss)
    defer ar.tensor_release(loss)

    //// Checked against PyTorch
    // >>> import torch
    // >>> import torch.nn.functional as F
    // >>> logits = torch.tensor([[2.0, -1.0, 0.5]], requires_grad=True).float()
    // >>> targets = torch.tensor([0])  # Class index 0
    // >>> # Compute loss
    // >>> loss = F.cross_entropy(logits, targets)
    // >>> loss.backward()
    // >>> print(f"Loss: {loss.item()}")
    // >>> print(f"Gradients: {logits.grad.tolist()}")
    expected_grad := []f32{-0.214402973651886, 0.03911257162690163, 0.17529039084911346}

    testing.expect(
        t,
        slice.equal(logits.grad.data, expected_grad),
        fmt.tprintf("Expected gradients %v, got %v", expected_grad, logits.grad.data),
    )
}
