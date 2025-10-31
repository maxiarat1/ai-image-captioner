# Async Database Architecture

## Overview

The application now uses asynchronous database operations to prevent DuckDB I/O from blocking AI model inference. This significantly improves throughput, especially during batch processing.

## Requirements

Async Flask support requires `flask[async]` which is now included in `requirements.txt`:
```bash
pip install 'flask[async]>=2.3.0'
```

This installs `asgiref` which enables async view functions in Flask.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Flask App                            │
│  ┌────────────┐              ┌──────────────────────┐       │
│  │   Sync     │              │   Async Endpoints    │       │
│  │ Endpoints  │              │  /generate           │       │
│  │ (CRUD)     │              │  /generate/batch     │       │
│  └────────────┘              └──────────────────────┘       │
│       │                               │                      │
│       ├───────────────────────────────┘                      │
│       │                                                      │
│  ┌────▼──────────────┐        ┌────────────────────────┐   │
│  │ SessionManager    │        │ AsyncSessionManager    │   │
│  │  (sync/blocking)  │        │  (async/non-blocking)  │   │
│  └───────────────────┘        └────────────────────────┘   │
│                                         │                    │
│                                         ▼                    │
│                              ┌────────────────────┐         │
│                              │   Thread Pool      │         │
│                              │  (4 DB workers)    │         │
│                              └────────────────────┘         │
│                                         │                    │
└─────────────────────────────────────────┼───────────────────┘
                                          │
                                          ▼
                                   ┌──────────────┐
                                   │   DuckDB     │
                                   │  (SQLite-    │
                                   │   style DB)  │
                                   └──────────────┘
```

## Key Components

### 1. **AsyncSessionManager** (`backend/database/async_session_manager.py`)

Async wrapper around DuckDB operations:

- **`get_image_path(image_id)`** - Async single image path lookup
- **`get_image_paths_batch(image_ids)`** - Async batch path lookup (concurrent)
- **`save_caption(image_id, caption)`** - Async caption save
- **`save_captions_batch(captions_data)`** - Async batch caption save (concurrent)

All methods use `run_in_thread()` to execute blocking DuckDB calls in a thread pool.

### 2. **Async Helpers** (`backend/utils/async_helpers.py`)

Utilities for concurrent execution:

- **`get_db_thread_pool()`** - Returns shared thread pool (4 workers)
- **`run_in_thread(func, *args)`** - Execute blocking function asynchronously
- **`async_db_operation`** - Decorator to convert sync DB functions to async
- **`gather_with_concurrency(n, *tasks)`** - Run tasks with concurrency limit

### 3. **Modified Endpoints** (`backend/app.py`)

#### `/generate` (Single Image)
```python
async def generate_caption():
    # 1. Async DB lookup (non-blocking)
    image_path = await async_session_manager.get_image_path(image_id)

    # 2. AI inference on GPU (blocking, but that's expected)
    caption = model_adapter.generate_caption(image, prompt, parameters)

    # 3. Fire-and-forget caption save (non-blocking)
    asyncio.create_task(async_session_manager.save_caption(image_id, caption))

    # 4. Return immediately without waiting for DB write
    return jsonify({"caption": caption, ...})
```

#### `/generate/batch` (Multiple Images)
```python
async def generate_captions_batch():
    # 1. Load all paths concurrently (single batch query)
    image_paths_dict = await async_session_manager.get_image_paths_batch(image_ids)

    # 2. AI inference on GPU (blocking for all images)
    captions = model_adapter.generate_captions_batch(images, prompts, parameters)

    # 3. Save all captions concurrently (fire-and-forget)
    asyncio.create_task(async_session_manager.save_captions_batch(captions_data))

    # 4. Return immediately
    return jsonify({"results": results, ...})
```

## Performance Benefits

### Before (Synchronous)
```
Single Image Processing:
├─ DB read image path     [10ms] ◄─── blocks thread
├─ AI inference           [2000ms] ◄─── blocks thread
└─ DB save caption        [15ms] ◄─── blocks thread
Total: ~2025ms per image
```

### After (Asynchronous)
```
Single Image Processing:
├─ DB read image path     [10ms] ◄─── async (thread pool)
├─ AI inference           [2000ms] ◄─── GPU thread
└─ DB save caption        [~0ms] ◄─── fire-and-forget, runs in background
Total: ~2010ms per image (15ms faster + non-blocking saves)

Batch Processing (10 images):
├─ DB read all paths      [30ms] ◄─── single concurrent batch query
├─ AI inference (batch)   [5000ms] ◄─── GPU processes all
└─ DB save all captions   [~0ms] ◄─── fire-and-forget batch save
Total: ~5030ms for 10 images vs ~20,250ms before (4x faster)
```

## Thread Safety

- Each thread in the pool gets its own DuckDB connection via `_get_connection()`
- DuckDB connections are NOT shared between threads
- Fire-and-forget tasks complete in background without blocking responses
- Original `SessionManager` remains unchanged for backward compatibility

## Usage

### For New Code (Async)
```python
from database import AsyncSessionManager

async_manager = AsyncSessionManager()

# In async endpoint
image_path = await async_manager.get_image_path(image_id)
await async_manager.save_caption(image_id, caption)
```

### For Legacy Code (Sync)
```python
from database import SessionManager

manager = SessionManager()

# In sync endpoint
image_path = manager.get_image_path(image_id)
manager.save_caption(image_id, caption)
```

## Configuration

Thread pool size can be adjusted in `utils/async_helpers.py`:
```python
_DB_POOL_SIZE = 4  # Increase for more concurrent DB operations
```

## Future Improvements

1. **Connection Pooling**: Reuse DuckDB connections instead of creating new ones
2. **Batch Optimizations**: Use DuckDB's `executemany()` for batch inserts
3. **Async Model Loading**: Make model loading async to prevent startup blocking
4. **Progress Tracking**: Add real-time progress updates for batch operations
5. **Error Handling**: Implement retry logic for failed background saves

## Monitoring

Enable debug logging to see async operation timing:
```python
logging.getLogger('asyncio').setLevel(logging.DEBUG)
```

Log messages will show:
- Thread pool initialization
- Background task completion
- DB operation timing

## Testing

Test async endpoints with concurrent requests:
```bash
# Single image (async)
curl -X POST http://localhost:5000/generate \
  -F "image_id=abc123" \
  -F "model=blip"

# Batch processing (async)
curl -X POST http://localhost:5000/generate/batch \
  -H "Content-Type: application/json" \
  -d '{"image_ids": ["id1", "id2", "id3"], "model": "blip"}'
```

## Backward Compatibility

All existing endpoints using `SessionManager` continue to work:
- `/session/register-folder`
- `/session/register-files`
- `/images`
- `/session/clear`
- etc.

Only inference endpoints (`/generate`, `/generate/batch`) use async operations.
