# SurRoL Environment Parameters

This document lists the key environment and simulation parameters extracted from the original SurRoL codebase (`archive/SurRoL/surrol/gym/surrol_env.py`). This serves as a reference for校对 our custom `dvrk_gym` implementation.

## 1. Rendering Parameters

### Render Dimensions
- **RENDER_HEIGHT**: `480`
- **RENDER_WIDTH**: `640`

### Camera View Matrix (`computeViewMatrixFromYawPitchRoll`)
- **cameraTargetPosition**: `(0, 0, 0.2)`
- **distance**: `1.5`
- **yaw**: `90`
- **pitch**: `-36`
- **roll**: `0`
- **upAxisIndex**: `2`

### Camera Projection Matrix (`computeProjectionMatrixFOV`)
- **fov**: `45`
- **aspect**: `640 / 480`
- **nearVal**: `0.1`
- **farVal**: `20.0`

## 2. Physics and Scene Parameters

- **Light Position**: `(10.0, 0.0, 10.0)`
- **Gravity**: `(0, 0, -9.81)`
- **Plane URDF Position**: `(0, 0, -0.001)`

## 3. Simulation Parameters

- **Step Duration (`_duration`)**: `0.2`
