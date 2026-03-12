# Football Commentary AI System - Architecture Overview

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      VIDEO INPUT SOURCES                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ Video File   │  │  Webcam (0)  │  │ IP/RTSP Stream       │  │
│  │ (*.mp4)      │  │  USB Camera  │  │ (broadcast/stadium)  │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                  FRAME CAPTURE & RESIZE (30 FPS)                │
│                    1280x720 resolution                           │
└────────────────────────┬────────────────────────────────────────┘
                         │
          ┌──────────────┴──────────────┐
          │                             │
          ▼                             ▼
    ┌──────────────┐          ┌──────────────────────┐
    │ YOLO v8      │          │ Game State Tracking  │
    │ Object       │          │ ├─ Score            │
    │ Detection    │          │ ├─ Possession       │
    │ ├─ Players   │          │ ├─ Minute           │
    │ ├─ Ball      │          │ ├─ Period           │
    │ ├─ Actions   │          │ └─ Recent Events    │
    │ └─ Teamwork  │          └──────────────────────┘
    └──────┬───────┘
           │
    ┌──────▼──────────────────────────┐
    │  MEANINGFUL ACTION FILTERING     │
    │  (Avoid commentary for every     │
    │   frame - only important events) │
    └──────┬──────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────┐
│              CONTEXT-AWARE COMMENTARY GENERATION                 │
│                                                                  │
│  ┌──────────────────┐  ┌──────────────────┐  ┌─────────────┐  │
│  │  OpenAI GPT-4    │  │  Local Ollama    │  │ Hugging     │  │
│  │  (Cloud, best)   │  │  (Private, fast) │  │ Face Models │  │
│  └──────────────────┘  └──────────────────┘  └─────────────┘  │
└────────────────────────┬────────────────────────────────────────┘
                         │
    Generates 1-2 sentences of fresh, authentic commentary
    considering:
    - Current game state (score, minute, possession)
    - Player/action details (who, what, where)
    - Match context (is this a crucial moment?)
    - Commentary history (avoid repeating)
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    TEXT-TO-SPEECH (TTS)                          │
│                Microsoft Edge TTS (cloud)                        │
│                Peter Drury voice style                           │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    AUDIO PLAYBACK                                │
│                  Synchronized with video                         │
└─────────────────────────────────────────────────────────────────┘
```

## Component Descriptions

### 1. **Object Detection (YOLOv8)**
- Real-time player and ball detection
- Team classification via jersey color analysis
- Player tracking across frames
- Action inference (shooting, passing, defending, etc.)
- ~100ms per frame on GPU, ~500ms on CPU

### 2. **Commentary Generation (LLMs)**
- **OpenAI GPT-4**: Cloud-based, highest quality, requires API key ($)
- **Ollama (Mistral)**: Local, free, private, fast, good quality
- **Hugging Face**: Local, various model sizes, flexible but requires tuning
- Generates contextual, varied commentary based on game state

### 3. **Voice Synthesis (Edge TTS)**
- Converts text to speech using Microsoft/Azure TTS
- Peter Drury style analysis commentary
- ~2-3 seconds per sentence
- Cloud-dependent (requires internet)

### 4. **Game State Management**
- Tracks score, time, possession
- Maintains event history
- Updates based on detected actions
- Provides context to commentary generator

## Data Flow Example

```
Minute 45:
┌─────────────────────┐
│ FRAME INPUT         │
│ Ball near goal      │
│ Striker advancing   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ DETECTION           │
│ Action: "shooting"  │
│ Player: #9          │
│ Confidence: 0.92    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────┐
│ GAME STATE             │
│ Home: 1, Away: 0       │
│ Possession: Home       │
│ Late game (45')        │
└──────────┬─────────────┘
           │
           ▼
┌──────────────────────────────────────┐
│ LLM GENERATION (GPT-4)               │
│ Input: Player #9 shooting in box,    │
│ 1-0 down, possession home team       │
│ Output:                              │
│ "The striker strikes from close!     │
│  Saved by the keeper! Brilliant!"    │
└──────────┬───────────────────────────┘
           │
           ▼
┌──────────────────────────────────────┐
│ TEXT-TO-SPEECH                       │
│ "The striker strikes from close!     │
│  Saved by the keeper! Brilliant!"    │
└──────────┬───────────────────────────┘
           │
           ▼
┌──────────────────────────────────────┐
│ AUDIO PLAYBACK (2 seconds)           │
│ [SPEAKER]: Commentary audio plays    │
└──────────────────────────────────────┘
```

## Performance Metrics

| Component | GPU | CPU | Bottleneck |
|-----------|-----|-----|-----------|
| YOLO Detection | ~100ms | ~500ms | Real-time capable |
| Commentary Gen (OpenAI) | ~1500ms | ~1500ms | Network latency |
| Commentary Gen (Ollama) | ~800ms | ~3000ms | Model inference |
| TTS Generation | ~2000ms | ~2000ms | Cloud service |
| Total (end-to-end) | ~4.4s | ~6s | TTS is bottleneck |

**FPS achievable**: 20-30 FPS on GPU, 5-10 FPS on CPU

## Deployment Options

### Local (Development)
- Run all components on personal machine
- Best for: Development, testing, single-match analysis
- Requirements: 8GB RAM, GPU optional

### Cloud (AWS/Google Cloud)
- YOLOv8 on GPU instance
- OpenAI API for commentary
- Scalable to multiple streams
- Best for: Professional streaming, multiple matches

### Edge (Jetson/Pi)
- Full system on-device
- Low latency, no cloud dependency
- Best for: Stadium deployment, real-time local system

### Hybrid
- Detection on edge (local)
- Commentary on cloud (better quality)
- Balance of performance and quality

## Key Improvements Over Template-Based System

| Aspect | Template System | ML System |
|--------|-----------------|-----------|
| Variety | ~50 templates per action | Infinite, unique commentary |
| Context Awareness | Limited context | Full game state context |
| Natural Language | Repetitive | Natural, authentic |
| Personalization | Generic | Can be fine-tuned to style |
| Adaptability | Fixed rules | Learns from data |
| Commentary Quality | Good | Excellent (GPT-4) to Good (Ollama) |

## Next Steps for Production

1. **Train custom YOLO model** on your team's footage
   - See TRAINING_GUIDE.md
   
2. **Collect training data** for commentary fine-tuning
   - Transcribe professional commentary
   - Align with video timestamps
   - See TRAINING_GUIDE.md

3. **Deploy to production** (Docker, cloud, or edge)
   - See DEPLOYMENT_GUIDE.md

4. **Monitor & optimize** performance
   - Track FPS, latency, quality
   - Adjust model sizes as needed

5. **Integrate with broadcast software**
   - OBS streaming
   - TV broadcast systems
   - Social media platforms
