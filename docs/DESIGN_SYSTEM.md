# RATISS VEGA — Design System

## Brand Story

**Vega** is the calibration star. Every other star's brightness is measured against Vega. RATISS VEGA is the same: not the biggest model, but the **reference for frugal, verifiable intelligence on CPU**.

Taglines:
- *Thinking in Constellations*
- *The Calibration Engine*
- *From paths to constellations*

## Logo Construction

- Grid: 1024×1024, circular spiral centered
- Spiral: 48 arms (one per amorce from RATISS-X), tapering 12px → 2px
- Center glow: radial gradient `#FFFFFF` 0% → `#B185DB` 40% → transparent 80%
- Constellation: Lyra pattern, 9 stars connected by 1px `#F8F9FA` lines, each star 6px + glow
- Typography: custom geometric sans, tracking 12%, weight 700

## Iconography

- **GPS pin** with star inside (routing)
- **Mixer** — three interlocking orbits
- **Chip** — CPU with Vega star etched
- **Seal** — SHA-256 badge with checkmark

## Spacing & Radius

- Scale: 4, 8, 12, 16, 24, 32, 48
- Card radius: 16px
- Button radius: 12px
- Glass blur: 12–16px

## Motion

- Ease: `cubic-bezier(0.16, 1, 0.3, 1)` (smooth space drift)
- Path animation: dash offset 1.2s infinite linear
- Star twinkle: opacity 0.6→1.0, 2s ease-in-out infinite alternate
- Panel entrance: translateY 8px → 0, 300ms

## Assets Delivered

- `ratiss-vega-logo.png` (1024×1024, AI-generated, selected Option 2)
- `ratiss-vega-banner.png` (1920×600, cinematic header)
- `architecture-hologram.png` (holographic blueprint)
- `ui-dashboard.png` (dashboard mockup)
