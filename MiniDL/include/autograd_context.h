#pragma once

struct AutogradContext {
    static bool grad_enabled;

    static void set_grad_enabled(bool val) {
        grad_enabled = val;
    }
};
