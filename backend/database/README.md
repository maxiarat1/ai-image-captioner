# Database Package

Clean, organized database operations with async support for non-blocking AI inference.

## Structure

```
database/
├── __init__.py              # Clean exports
├── session_manager.py       # Original synchronous operations
├── async_session_manager.py # New async operations
└── README.md               # This file
```

## Quick Start

### Import
```python
from database import SessionManager, AsyncSessionManager
```

### Synchronous (Legacy, Simple CRUD)
```python
session = SessionManager()
path = session.get_image_path(image_id)
session.save_caption(image_id, caption)
```

### Asynchronous (AI Inference Paths)
```python
async_session = AsyncSessionManager()

# In async endpoint
path = await async_session.get_image_path(image_id)
await async_session.save_caption(image_id, caption)

# Fire-and-forget (non-blocking save)
asyncio.create_task(async_session.save_caption(image_id, caption))
```

## When to Use Which

### Use `SessionManager` (Sync) for:
- Simple CRUD operations
- Synchronous endpoints
- Quick reads/writes where blocking is acceptable
- Legacy code compatibility

### Use `AsyncSessionManager` (Async) for:
- AI inference paths (`/generate`, `/generate/batch`)
- Long-running operations
- Batch processing with concurrent DB operations
- When you need non-blocking DB I/O

## Key Benefits of Async

1. **Non-blocking**: DB operations don't block AI inference
2. **Concurrent**: Multiple DB operations run in parallel
3. **Fire-and-forget**: Save operations can happen in background
4. **Better throughput**: Especially for batch operations

## Available Methods

### SessionManager (Sync)
- `get_image_path(image_id)` - Get single image path
- `get_image_paths_batch(image_ids)` - Fetch multiple paths in one query
- `save_caption(image_id, caption)` - Save single caption
- `save_captions_batch(captions_data)` - Save multiple captions in one transaction
- `register_folder(folder_path)` - Register all images in folder
- `register_files(file_metadata_list)` - Pre-register files
- `list_images(page, per_page, search)` - Paginated list
- `clear_all()` - Clear all images
- `delete_image(image_id)` - Delete single image
- `get_image_metadata(image_id)` - Get metadata
- `save_uploaded_file(image_id, file_data, filename)` - Save upload

### AsyncSessionManager (Async)
- `async get_image_path(image_id)` - Get single image path (async)
- `async get_image_paths_batch(image_ids)` - Get multiple paths (concurrent)
- `async save_caption(image_id, caption)` - Save single caption (async)
- `async save_captions_batch(captions_data)` - Save multiple captions (concurrent)
- `async get_image_metadata(image_id)` - Get metadata (async)
- `async list_images(page, per_page, search)` - Paginated list (async)
- `shutdown()` - Graceful thread pool shutdown (delegates to async_helpers)

> **Note:** The async manager now delegates to `SessionManager` for all
> operations and simply runs them inside the shared thread pool. SQL only
> lives in one place, so sync and async paths stay consistent automatically.

## Performance Example

**Batch processing 100 images:**

```
Synchronous:
├─ Load 100 paths:  100 × 10ms  = 1,000ms
├─ AI inference:                 = 20,000ms
└─ Save 100 captions: 100 × 15ms = 1,500ms
Total: 22,500ms

Asynchronous:
├─ Load 100 paths (concurrent):  = 50ms
├─ AI inference:                 = 20,000ms
└─ Save 100 captions (background)= ~0ms (non-blocking)
Total: 20,050ms (10% faster, response immediate)
```

## Thread Safety

- Async operations share the global DB thread pool (4 workers by default)
- Each SessionManager call opens its own DuckDB connection
- No shared connection state
- Safe for concurrent operations

## See Also

- `backend/utils/async_helpers.py` - Async utility functions
- `backend/ASYNC_ARCHITECTURE.md` - Full architecture documentation
