Model-Based Optical Metrology – Interview Preparation Notes
1. Fundamental Physics Layer

All optical metrology systems are fundamentally governed by electromagnetic wave propagation and boundary interactions.

At the core:

Light interacts with material through dielectric response (n, k).

Boundary conditions determine reflection, transmission, and phase change.

Optical response encodes structural information.

Industrial implication:

We are not just measuring light intensity — we are extracting structural parameters from electromagnetic scattering behavior.

2. Thin Film and Multilayer Systems

In semiconductor metrology, multilayer stacks are common.

Key principles:

Optical response is determined by phase accumulation inside layers.

Thickness variations change phase, which alters interference patterns.

Multilayer systems can amplify or suppress spectral features.

Engineering perspective:

Thin film design is essentially phase engineering.
Measurement sensitivity depends on how strongly phase varies with geometry.

3. Scatterometry and Periodic Structures

For periodic structures:

Diffraction encodes geometry information.

Spectral response depends on CD, height, sidewall angle, pitch.

Forward models (e.g., RCWA) compute diffraction efficiency.

Industrial mindset:

We never measure geometry directly.
We measure optical response and invert the model.

4. Ellipsometry and Phase-Sensitive Measurement

Ellipsometry increases information density by measuring phase differences.

Why this matters:

Intensity-only measurements may be insufficient.

Phase sensitivity improves parameter identifiability.

Inverse fitting becomes better conditioned.

This is why phase-based metrology is powerful in advanced nodes.

5. Model-Based Inverse Problem

The core industrial workflow:

Parameterize structure

Build forward EM solver

Simulate spectrum

Compare with measured data

Optimize parameters

Estimate uncertainty

Key insight:

Metrology accuracy is not determined only by model accuracy,
but also by parameter correlation and sensitivity.

6. Sensitivity and Identifiability

Important interview theme:

Not every parameter is measurable.

If two parameters produce nearly identical spectral change,
they are highly correlated.

Good metrology system design requires:

High sensitivity to key parameters

Low correlation between parameters

Stable inversion

This is where system-level thinking becomes critical.

7. Error Budget Thinking

Industrial optical systems must consider:

Optical noise

Mechanical drift

Thermal expansion

Detector noise

Contamination

Long-term aging

Metrology performance is a system property, not a single component property.

8. DUV / High-Power Considerations

In DUV systems:

Material absorption increases

Laser damage threshold matters

Long-term degradation occurs

Optical contamination becomes critical

Industrial priority:

Stability over lifetime > peak performance.
