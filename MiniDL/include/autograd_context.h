#pragma once

struct AutogradContext {
    static bool grad_enabled;   // grad_enabled is a static boolean flag.
    // static -> only one shared copy of this variable across the entire program, not per instance.
    // It acts like a global setting: “Are we tracking gradients right now?”
    static void set_grad_enabled(bool val) {
        grad_enabled = val;
    }
    // Provides a static method to turn gradient tracking on or off.
};
