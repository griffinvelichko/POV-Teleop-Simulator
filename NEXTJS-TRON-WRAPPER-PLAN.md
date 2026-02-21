# NextJS TRON-Themed Wrapper Plan for POV Teleop Simulator

**Date:** February 21, 2026
**Status:** Verified & Corrected — Implementation Ready
**Goal:** Wrap the gym-soarm robotic arm simulator in a NextJS app with an authentic TRON-themed, futuristic UI while ensuring the sim runs perfectly.

---

## Verification Summary (Feb 21, 2026)

Issues found and corrected during feasibility review:

| Issue | Severity | Fix |
|-------|----------|-----|
| `camera_config='front_wrist'` has a KeyError bug in gym-soarm v0.4.0 | **Critical** | Changed to `diagonal_wrist` (default). Updated all pixel key references. |
| `pixels_agent_pos` listed as a render mode | Medium | Clarified: it's an `obs_type`, not a render mode. Render modes are `human` and `rgb_array` only. |
| Pixel dict keys listed as `front_camera`/`wrist_camera` | **Critical** | Corrected to `diagonal`/`wrist.right` for `diagonal_wrist` config. |
| `fastapi-frame-stream` recommended | Medium | Removed — abandoned since 2022. Use raw `StreamingResponse`. |
| `cv-ws-streamer` recommended | Medium | Removed — empty placeholder package with no code. |
| "Tailwind Cyberpunk Components" (12+ components) referenced | Medium | Removed — does not exist as an npm package (hallucinated). |
| "Neon-Nexus" React package referenced | Medium | Removed — does not exist on npm (hallucinated). |
| No warning about `next/image` breaking MJPEG | **Critical** | Added `MjpegStream` component pattern using `useRef` + `memo`. |
| No mention of iOS MJPEG incompatibility | High | Added browser compatibility table + WebSocket/canvas fallback. |
| `env.step()` called in async without threading | High | Added `asyncio.to_thread()` pattern for blocking MuJoCo calls. |

---

## Executive Summary

This plan documents deep research into (1) the gym-soarm MuJoCo simulator, (2) streaming Python-rendered video to a NextJS frontend, (3) TRON-themed UI/UX libraries outside the typical AI aesthetic, and (4) architecture to ensure the robotic arm simulation works flawlessly within the web wrapper.

**Key Finding:** The simulator must run in a **Python backend** (FastAPI). The browser displays **streamed frames** via MJPEG or WebSocket. A pure browser-based MuJoCo WASM approach is possible but requires significant porting of the SO-ARM101 model and is not recommended for the first iteration.

---

## Part 1: Simulator Deep Dive (gym-soarm)

### 1.1 What We're Using

| Property | Value |
|----------|-------|
| **Package** | gym-soarm ≥0.4.0 |
| **Physics** | MuJoCo 2.3.7 via dm-control |
| **Robot** | SO-ARM101 (6DOF + gripper) |
| **API** | Gymnasium (OpenAI Gym compatible) |
| **Rendering** | OpenCV viewer (human) or pixel arrays (headless) |

### 1.2 Observation Structure

**WARNING: `camera_config='front_wrist'` is BROKEN in gym-soarm v0.4.0** due to a naming mismatch between `env.py` and `tasks/sim.py` (the env requests key `"wrist_camera"` but the task stores it under `"wrist.right"`, causing a `KeyError`). Use `camera_config='diagonal_wrist'` (the default) or `'front_only'` instead.

With `obs_type='pixels_agent_pos'` and `camera_config='diagonal_wrist'`:

```python
obs = {
    'agent_pos': np.ndarray(shape=(6,)),      # Joint positions in radians
    'pixels': {
        'diagonal': np.ndarray(480, 640, 3),     # RGB, uint8
        'wrist.right': np.ndarray(480, 640, 3),  # RGB, uint8
    }
}
```

With `camera_config='front_only'`:

```python
obs = {
    'agent_pos': np.ndarray(shape=(6,)),
    'pixels': {
        'front_camera': np.ndarray(480, 640, 3),  # RGB, uint8
    }
}
```

**Camera config compatibility matrix (v0.4.0):**

| Config | Pixel Keys | Works? |
|--------|-----------|--------|
| `'front_only'` | `['front_camera']` | Yes |
| `'diagonal_wrist'` (default) | `['diagonal', 'wrist.right']` | Yes |
| `'front_wrist'` | `['front_camera', 'wrist_camera']` | **NO — KeyError** |
| `'all'` | `['overview_camera', 'front_camera', 'front_close', 'wrist_camera']` | **NO — KeyError** |

**Critical:** The `obs['pixels']` dict contains **ready-to-stream** RGB numpy arrays. No separate `render()` call is needed when using `pixels_agent_pos` — each `env.step(action)` returns fresh camera frames.

### 1.3 Action Space

```python
action = np.array([
    shoulder_pan,   # rad
    shoulder_lift,  # rad
    elbow_flex,     # rad
    wrist_flex,     # rad
    wrist_roll,     # rad
    gripper,        # 0.0–1.0 normalized
])
```

### 1.4 Render Modes & Headless

`render_mode` and `obs_type` are **separate parameters** — do not confuse them.

**`render_mode`** controls what `env.render()` does:

| `render_mode` | Use Case | Behavior |
|------|----------|--------|
| `'human'` | Local dev with OpenCV window | Opens native OpenCV window with keyboard controls (1-5 switch cameras, q/ESC quit) |
| `'rgb_array'` | Headless / server | `env.render()` returns numpy array; **recommended for backend streaming** |

**`obs_type`** controls what `env.step()` and `env.reset()` return:

| `obs_type` | Returns |
|------|----------|
| `'pixels_agent_pos'` | `dict` with `'pixels'` (camera images) + `'agent_pos'` (6D joint positions) |
| `'pixels'` | Camera images only |
| `'agent_pos'` | Joint positions only |

**For the web wrapper, use `render_mode='rgb_array'` with `obs_type='pixels_agent_pos'`.** Pixel observations are returned by `step()`/`reset()` regardless of render mode. The `render_mode` only affects the explicit `env.render()` call — do NOT pass `render_mode=None` as it will crash if `render()` is ever called.

For **headless server** (e.g., Docker, cloud):

- **Linux:** `export MUJOCO_GL=egl` (EGL for GPU headless)
- **macOS:** `export MUJOCO_GL=glfw` (glfw works with display)
- **Docker:** Install `libglew2.1 libgl1-mesa-glx libosmesa6`; ensure GPU device access

### 1.5 Camera Configurations

- `front_only` — Single camera, fastest, key: `front_camera`
- `diagonal_wrist` — Diagonal + wrist (actual default), keys: `diagonal`, `wrist.right`
- ~~`front_wrist`~~ — **BROKEN in v0.4.0** (KeyError bug)
- ~~`all`~~ — **BROKEN in v0.4.0** (KeyError bug)

### 1.6 Dependencies (from gym-soarm)

- gymnasium ≥0.29.1
- mujoco ≥2.3.7, <3.0.0
- dm-control ≥1.0.14
- imageio[ffmpeg] ≥2.34.0

---

## Part 2: Streaming Architecture — Python to NextJS

### 2.1 Why Not Run Sim in Browser?

| Approach | Pros | Cons |
|----------|------|------|
| **MuJoCo WASM** (zalo/mujoco_wasm) | Pure browser, no backend | SO-ARM101 model + STL meshes must be ported; no gym-soarm; complex |
| **Viser** | Web-based 3D, Python API | Visualization only — no physics; URDF vs MJCF mismatch |
| **Python + Stream** | gym-soarm works as-is; full pipeline | Requires backend process |

**Verdict:** Python backend with frame streaming is the only path that preserves the existing gym-soarm pipeline and guarantees the sim works perfectly.

### 2.2 Streaming Options

| Method | Latency | Complexity | Browser Support |
|--------|---------|------------|-----------------|
| **MJPEG** (`multipart/x-mixed-replace`) | ~50–100ms | Low | `<img src="...">` works natively |
| **WebSocket + base64 JPEG** | ~30–80ms | Medium | Full control, bidirectional |
| **WebSocket + binary** | ~25–70ms | Medium | Most efficient |

**Recommendation:** Start with **MJPEG** for sim and camera feeds — trivial to implement, works with a simple `<img>` tag. Add WebSocket later for bidirectional control (e.g., manual overrides, joint sliders) if needed.

### 2.3 MJPEG Implementation (FastAPI)

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import cv2
import asyncio
import numpy as np

app = FastAPI()

# Sim step is blocking (MuJoCo) — run in a thread to avoid blocking the event loop
async def generate_sim_frames():
    while sim_running:
        # Run the blocking sim step in a thread
        obs = await asyncio.to_thread(env.step, current_action)
        obs = obs[0]  # env.step returns (obs, reward, terminated, truncated, info)
        frame = obs['pixels']['diagonal']  # RGB 480x640x3 (use 'diagonal' for diagonal_wrist config)
        _, jpeg = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, 85])
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n'
            + jpeg.tobytes()
            + b'\r\n'
        )
        await asyncio.sleep(0.033)  # ~30 FPS cap

@app.get("/stream/sim")
async def sim_stream():
    return StreamingResponse(
        generate_sim_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
```

**Gotchas:**
- `env.step()` and `cv2.imencode()` are **blocking calls** — use `asyncio.to_thread()` or a dedicated capture thread feeding an `asyncio.Queue` to avoid stalling the FastAPI event loop.
- When the client disconnects, the generator may continue running. Use a `try/finally` block to clean up.
- FastAPI's Swagger UI (`/docs`) cannot render MJPEG streams — test with a browser directly.
- Each MJPEG stream holds an open HTTP connection. Browsers limit to ~6 concurrent connections per origin.

**Do NOT use third-party MJPEG libraries.** `fastapi-frame-stream` is abandoned (no releases since 2022). `cv-ws-streamer` is an empty placeholder package. `labthings_fastapi` is a full lab instrument framework — overkill. The raw `StreamingResponse` pattern above (~15 lines) is the correct approach.

### 2.4 Full Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FASTAPI BACKEND (port 8000)                          │
│                                                                              │
│  Camera ──→ Pose ──→ JointMapper ──→ Smoother ──→ action                    │
│    │          │           │              │              │                    │
│    │          │           │              │              ▼                    │
│    │          │           │              │         gym-soarm                 │
│    │          │           │              │         env.step(action)         │
│    │          │           │              │              │                    │
│    ▼          ▼           │              │              ▼                    │
│  MJPEG     (optional)     │              │         obs['pixels']              │
│  /stream/camera           │              │              │                    │
│                           │              │              ▼                    │
│                           │              │         MJPEG /stream/sim         │
└───────────────────────────┼──────────────┼─────────────────────────────────┘
                            │              │
                            └──────────────┘
```

### 2.5 NextJS Integration

**CRITICAL: Do NOT use `next/image` for MJPEG streams.** The `<Image>` component applies optimization (resizing, format conversion) that breaks live streams, and its `onError` handler causes reload loops. Use a native `<img>` element with `useRef` + `React.memo`:

```tsx
import { useRef, useEffect, memo } from "react";

interface MjpegStreamProps {
  src: string;
  alt?: string;
  className?: string;
}

const MjpegStream = memo(function MjpegStream({ src, alt = "Stream", className }: MjpegStreamProps) {
  const imgRef = useRef<HTMLImageElement>(null);

  useEffect(() => {
    // Set src via ref to avoid React re-render interference
    if (imgRef.current) {
      imgRef.current.src = src;
    }
    return () => {
      if (imgRef.current) {
        imgRef.current.src = "";  // Tear down HTTP connection on unmount
      }
    };
  }, [src]);

  return (
    // eslint-disable-next-line @next/next/no-img-element
    <img ref={imgRef} alt={alt} className={className} />
  );
});

export default MjpegStream;
```

**Why `useRef` + `memo`?** Any React state change that re-renders the component would reset the `<img>` src attribute, restarting the MJPEG HTTP connection (visible flicker + new connection each time). Setting `src` imperatively via ref prevents this.

**Integration options:**
- **Same-origin proxy (recommended):** NextJS rewrites `/api/stream/*` to FastAPI — avoids CORS entirely. Add to `next.config.js`:
  ```js
  async rewrites() {
    return [{ source: '/api/stream/:path*', destination: 'http://localhost:8000/stream/:path*' }];
  }
  ```
- **Cross-origin:** Set `NEXT_PUBLIC_API_URL=http://localhost:8000`. FastAPI must set `Access-Control-Allow-Origin`.

**Browser compatibility for MJPEG `<img src>`:**

| Browser | Works? |
|---------|--------|
| Chrome (desktop) | Yes |
| Firefox (desktop) | Yes |
| Safari (macOS) | Yes (minor bug: setting `src=""` may not fully release connection) |
| Edge (Chromium) | Yes |
| **iOS (all browsers)** | **NO — MJPEG `<img>` streams do not work on iOS** |

If iOS support is needed, use a WebSocket approach (fetch frames as binary, paint to `<canvas>`).

---

## Part 3: TRON-Themed UI/UX — Beyond AI Slop

### 3.1 Design Principles

| Principle | Implementation |
|-----------|----------------|
| **Deep black** | `#0a0a0f` or `#050508` — not gray |
| **Neon accents** | Cyan `#00ffff`, orange `#ff6600`, magenta `#ff00ff` |
| **Grid** | Subtle horizontal/vertical lines, perspective grid |
| **Glow** | Layered `text-shadow` and `box-shadow` |
| **Typography** | Orbitron, Audiowide, or "Share Tech Mono" — avoid Inter |

### 3.2 Typography (Avoid Generic Fonts)

- **Headings:** [Orbitron](https://fonts.google.com/specimen/Orbitron) — geometric, futuristic
- **Body/UI:** [Share Tech Mono](https://fonts.google.com/specimen/Share+Tech+Mono) — monospace, terminal feel
- **Accent:** [Audiowide](https://fonts.google.com/specimen/Audiowide) — classic TRON

### 3.3 CSS Neon Effects (Reference)

```css
/* Layered neon text */
.neon-cyan {
  color: #00ffff;
  text-shadow:
    0 0 5px #00ffff,
    0 0 10px #00ffff,
    0 0 20px #00ffff,
    0 0 40px #00ffff;
}

/* Glowing border */
.neon-border {
  box-shadow:
    0 0 5px #00ffff,
    inset 0 0 5px rgba(0, 255, 255, 0.1);
  border: 1px solid #00ffff;
}
```

### 3.4 Library Stack (Outside Typical AI Stack)

| Category | Library | Status | Purpose |
|----------|---------|--------|---------|
| **Animation** | [Framer Motion](https://www.framer.com/motion/) | Real, widely used | Glow pulse, entrance animations |
| **3D** | [React Three Fiber](https://docs.pmnd.rs/react-three-fiber) + [@react-three/drei](https://github.com/pmndrs/drei) | Real, widely used | TRON grid floor, bloom post-processing |
| **Effects** | [motion-primitives GlowEffect](https://motion-primitives.com/docs/glow-effect) | Real, 5.3k stars | Configurable glow (rotate, pulse, breathe, colorShift). Install: `npx motion-primitives@latest add glow-effect` |
| **Components** | [shadcn/ui](https://ui.shadcn.com/) + community cyberpunk CSS vars from [shadcn.io](https://www.shadcn.io/theme/cyberpunk) | Real (note: shadcn.io is a third-party community site, not official shadcn/ui) | Copy-paste CSS variable overrides for neon palette |
| **Cyberpunk theme** | [daisyUI](https://daisyui.com) `data-theme="cyberpunk"` | Real, popular Tailwind plugin | Built-in cyberpunk theme with neon colors — alternative to custom CSS vars |
| **Styling** | Custom CSS variables + Tailwind | N/A | Full control; avoid default Tailwind palette |

### 3.5 What to Avoid

- Inter, system-ui, sans-serif
- Purple gradients (#8b5cf6, etc.)
- Rounded corners everywhere
- Generic "AI" aesthetic (gradient meshes, soft shadows)

### 3.6 TRON-Specific Visual Elements

- **Grid floor:** Perspective grid (Three.js or CSS gradient)
- **Scanlines:** `repeating-linear-gradient` overlay
- **Corner brackets:** Angular frames around panels
- **Data readouts:** Monospace font, cyan/amber on black

---

## Part 4: Implementation Plan

### Phase 1: Backend Streaming (FastAPI)

1. **Create `backend/` directory** with FastAPI app.
2. **Sim stream endpoint:** `/stream/sim` — MJPEG stream from `obs['pixels']['diagonal']` (using `diagonal_wrist` config).
3. **Teleop loop:** Run camera → pose → mapping → sim in a **dedicated thread** (not async — MuJoCo is blocking); feed frames to stream via `asyncio.Queue`.
4. **CORS:** Allow NextJS origin.
5. **Health check:** `/health` for readiness.

### Phase 2: NextJS Shell

1. **Create `frontend/`** with `npx create-next-app`.
2. **Sim viewer:** Use `MjpegStream` component (see Section 2.5) — NOT `next/image`.
3. **Camera viewer:** Optional second `MjpegStream` for camera feed split view.
4. **Environment:** `NEXT_PUBLIC_API_URL` or use NextJS rewrites for same-origin proxy.

### Phase 3: TRON UI

1. **Global styles:** CSS variables for colors, fonts.
2. **Layout:** Deep black background, neon-accented panels.
3. **Sim container:** Glowing border, scanline overlay.
4. **Dashboard:** Joint angles, FPS, status — monospace, cyan/amber.
5. **Optional:** React Three Fiber grid background.

### Phase 4: Polish

1. **Responsive:** Sim viewer scales; desktop-first (MJPEG doesn't work on iOS).
2. **Connection status:** Show "Connecting..." when stream unavailable. Use `onError` on the `<img>` ref to detect broken streams.
3. **Error handling:** Fallback UI for backend down.
4. **iOS fallback (stretch):** WebSocket + canvas-based frame display for mobile.

---

## Part 5: File Structure

```
pov-teleop/
├── backend/                    # FastAPI
│   ├── main.py                 # FastAPI app, routes
│   ├── stream.py               # MJPEG generators
│   ├── teleop_loop.py          # Camera → Pose → Sim (or import from src)
│   └── requirements.txt
├── frontend/                   # NextJS
│   ├── app/
│   │   ├── layout.tsx
│   │   ├── page.tsx
│   │   └── globals.css
│   ├── components/
│   │   ├── SimViewer.tsx
│   │   ├── CameraViewer.tsx
│   │   ├── Dashboard.tsx
│   │   └── NeonPanel.tsx
│   └── package.json
├── src/                        # Existing Python pipeline
│   ├── config.py
│   ├── camera.py
│   ├── pose.py
│   ├── mapping.py
│   ├── smoother.py
│   ├── sim.py
│   ├── display.py
│   └── main.py
└── NEXTJS-TRON-WRAPPER-PLAN.md
```

---

## Part 6: Critical Compatibility Checklist

| Item | Status | Notes |
|------|--------|-------|
| gym-soarm returns `obs['pixels']` from `step()` regardless of `render_mode` | ✅ Yes | Pixel observations are generated by `SoArmTask.get_observation()`, not by `render()` |
| `camera_config='diagonal_wrist'` works | ✅ Yes | Keys: `diagonal`, `wrist.right` |
| `camera_config='front_wrist'` works | ❌ **NO** | KeyError bug in v0.4.0 — naming mismatch |
| NumPy frames encodable to JPEG via `cv2.imencode` | ✅ Yes | Remember RGB→BGR conversion |
| MJPEG works in `<img src>` (desktop) | ✅ Yes | Chrome, Firefox, Safari, Edge |
| MJPEG works in `<img src>` (iOS) | ❌ **NO** | WebKit limitation — need WebSocket fallback |
| FastAPI `StreamingResponse` supports multipart | ✅ Yes | Use `asyncio.to_thread` for blocking calls |
| `next/image` works for MJPEG | ❌ **NO** | Use native `<img>` + `useRef` + `memo` |
| NextJS can load external stream URLs | ✅ Yes | Use rewrites proxy or CORS |
| MuJoCo headless (EGL) on Linux | ✅ Yes | `MUJOCO_GL=egl` |
| macOS local dev | ✅ Yes | `MUJOCO_GL=glfw` |

---

## Part 7: Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Stream latency too high | Use WebSocket + binary; reduce JPEG quality; reduce resolution (e.g., 320x240) |
| CORS issues | Proxy via NextJS rewrites (preferred) or configure FastAPI CORS middleware |
| GPU/EGL missing in Docker | Use `libosmesa6` software renderer |
| Backend crashes | Add `/health` endpoint; show "Reconnecting" in UI |
| MJPEG doesn't work on iOS | WebSocket + `<canvas>` fallback for mobile (stretch goal) |
| React re-renders break stream | Use `MjpegStream` component with `useRef` + `memo` (see Section 2.5) |
| `front_wrist` camera config crashes | Use `diagonal_wrist` (default) or `front_only` — see Section 1.2 |
| `env.step()` blocks async event loop | Run teleop loop in dedicated thread; use `asyncio.Queue` to pass frames to stream generator |

---

## Part 8: References

- [gym-soarm PyPI](https://pypi.org/project/gym-soarm/)
- [FastAPI StreamingResponse](https://fastapi.tiangolo.com/advanced/custom-response/#streamingresponse)
- [MJPEG multipart format](https://developer.mozilla.org/en-US/docs/Web/HTTP/MIME_types/multipart)
- [Viser](https://viser.studio/main/) — alternative visualization (no physics)
- [MuJoCo WASM](https://github.com/zalo/mujoco_wasm) — future browser option
- [CSS Neon Effects](https://css-tricks.com/how-to-create-neon-text-with-css/)
- [React Three Fiber](https://docs.pmnd.rs/react-three-fiber)
- [Framer Motion](https://www.framer.com/motion/)

---

## Appendix A: Quick Start Commands

```bash
# Terminal 1: Backend
cd backend && pip install -r requirements.txt && uvicorn main:app --reload --port 8000

# Terminal 2: Frontend
cd frontend && npm install && npm run dev

# Browser: http://localhost:3000
```

## Appendix B: TRON Color Palette

```css
:root {
  --tron-black: #0a0a0f;
  --tron-cyan: #00ffff;
  --tron-orange: #ff6600;
  --tron-magenta: #ff00ff;
  --tron-amber: #ffaa00;
  --tron-grid: rgba(0, 255, 255, 0.1);
}
```

---

## Appendix C: WebSocket Bidirectional Flow (Future / iOS Fallback)

When the full teleop pipeline runs on the backend, the browser only displays streams. WebSocket is needed for (a) iOS support (MJPEG doesn't work) and (b) manual overrides (joint sliders, keyboard control from web UI):

```python
import asyncio
from fastapi import FastAPI, WebSocket
import numpy as np

app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    current_action = np.zeros(6)

    async def send_frames():
        while True:
            # Run blocking sim.step in thread
            result = await asyncio.to_thread(env.step, current_action)
            obs = result[0]
            jpeg = frame_to_jpeg(obs['pixels']['diagonal'])  # See Appendix E
            await websocket.send_bytes(jpeg)
            await asyncio.sleep(1/30)  # ~30 FPS

    async def receive_actions():
        nonlocal current_action
        while True:
            data = await websocket.receive_json()
            current_action = np.array(data['action'])

    await asyncio.gather(send_frames(), receive_actions())
```

**Client-side (canvas renderer for iOS):**

```tsx
useEffect(() => {
  const ws = new WebSocket("ws://localhost:8000/ws");
  ws.binaryType = "arraybuffer";
  const canvas = canvasRef.current;
  const ctx = canvas?.getContext("2d");

  ws.onmessage = (event) => {
    const blob = new Blob([event.data], { type: "image/jpeg" });
    const url = URL.createObjectURL(blob);
    const img = new Image();
    img.onload = () => {
      ctx?.drawImage(img, 0, 0, canvas!.width, canvas!.height);
      URL.revokeObjectURL(url);
    };
    img.src = url;
  };

  return () => ws.close();
}, []);
```

---

## Appendix D: Package Research Summary

### TRON/Cyberpunk UI Packages (Verified Feb 2026)

| Package | Real? | Usable? | Notes |
|---------|-------|---------|-------|
| **shadcn/ui cyberpunk theme** ([shadcn.io](https://www.shadcn.io/theme/cyberpunk)) | Yes | Yes | Third-party community site (not official shadcn/ui). Copy-paste CSS variables. |
| **daisyUI** `cyberpunk` theme ([daisyui.com](https://daisyui.com)) | Yes | Yes | Real Tailwind plugin with built-in cyberpunk theme. `data-theme="cyberpunk"`. |
| **motion-primitives GlowEffect** ([motion-primitives.com](https://motion-primitives.com/docs/glow-effect)) | Yes | Yes | 5.3k GitHub stars. Modes: rotate, pulse, breathe, colorShift, flowHorizontal, static. |
| ~~**Tron-UI** (jasonsilvers)~~ | Yes | **No** | 2 GitHub stars, abandoned, based on Tachyons (not Tailwind). Do not use. |
| ~~**Tailwind Cyberpunk Components**~~ | **No** | **No** | Does not exist as an npm package. Was hallucinated. |
| ~~**Neon-Nexus**~~ | **No** | **No** | Does not exist as an npm package. Was hallucinated. |

### Video Streaming Packages (Verified Feb 2026)

| Package | Real? | Recommendation |
|---------|-------|----------------|
| ~~**fastapi-frame-stream**~~ | Yes but abandoned | **Do NOT use.** No releases since July 2022. Unnecessary SQLite overhead. |
| ~~**cv-ws-streamer**~~ | Empty placeholder | **Do NOT use.** No code, no docs, no source repo. |
| **labthings_fastapi** | Yes, active | Has `MJPEGStreamResponse` but the package is a full lab instrument framework — overkill. Copy the ~20-line pattern instead. |
| **Raw `StreamingResponse`** | Built into FastAPI | **USE THIS.** ~15 lines, no extra dependencies. See Section 2.3. |

### 3D/Animation (Researched)

| Package | Use Case |
|---------|----------|
| **React Three Fiber** | Declarative Three.js in React |
| **@react-three/drei** | Helpers: Grid, Bloom, Text3D |
| **Framer Motion** | Page/component animations, glow |

---

## Appendix E: Sim Frame Encoding (Exact Code)

gym-soarm returns **RGB** `uint8` arrays in `obs['pixels']`. OpenCV `imencode` expects **BGR**:

```python
import cv2
import numpy as np

def frame_to_jpeg(rgb_frame: np.ndarray, quality: int = 85) -> bytes:
    """Convert RGB numpy array to JPEG bytes for streaming."""
    bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buffer.tobytes()
```

**Pixel key reference (for use in `frame_to_jpeg` calls):**

```python
# camera_config='diagonal_wrist' (recommended)
diagonal_frame = obs['pixels']['diagonal']       # (480, 640, 3) RGB uint8
wrist_frame    = obs['pixels']['wrist.right']    # (480, 640, 3) RGB uint8

# camera_config='front_only'
front_frame    = obs['pixels']['front_camera']   # (480, 640, 3) RGB uint8
```
