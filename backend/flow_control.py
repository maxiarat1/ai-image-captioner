"""
Flow Control Hub - Centralized data routing and flow management.

This module provides a unified interface for managing how processed image data
(captions, tags, etc.) flows through the system. It decouples data generation
from data persistence and routing, creating a more modular architecture.

Key Features:
- Centralized routing decisions
- Order preservation for batch operations
- Flexible destination management
- Consistent error handling
- Easy to extend with new data flows
"""

import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import asyncio

logger = logging.getLogger(__name__)


# ============================================================================
# Data Flow Models
# ============================================================================

class FlowDestination(Enum):
    """Supported destinations for processed data."""
    DATABASE = "database"           # Save to persistent database
    MEMORY = "memory"              # Keep in memory only
    CALLBACK = "callback"          # Send to custom callback
    BROADCAST = "broadcast"        # Send to multiple destinations


@dataclass
class ProcessedData:
    """
    Container for processed image data with metadata.

    This standardized format ensures consistent data handling across the system.
    """
    image_id: str
    content: str                   # Caption, tags, or other processed text
    model_name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    sequence_num: int = 0          # For maintaining order in batch operations

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'image_id': self.image_id,
            'content': self.content,
            'model_name': self.model_name,
            'parameters': self.parameters,
            'metadata': self.metadata,
            'sequence_num': self.sequence_num
        }


@dataclass
class FlowConfig:
    """Configuration for data flow routing."""
    destination: FlowDestination = FlowDestination.DATABASE
    preserve_order: bool = True
    batch_size: int = 10           # For batched database writes
    callback: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Flow Control Hub
# ============================================================================

class FlowControlHub:
    """
    Centralized hub for managing data flow from processing to storage/output.

    This class acts as a single point of control for all data routing decisions,
    ensuring consistent behavior and making the architecture more maintainable.

    Example Usage:
        hub = FlowControlHub(async_session_manager)

        # Single data point
        await hub.route_data(ProcessedData(
            image_id="img_001",
            content="A beautiful sunset",
            model_name="blip"
        ))

        # Batch with order preservation
        await hub.route_batch([data1, data2, data3])
    """

    def __init__(self, async_session_manager=None):
        """
        Initialize the flow control hub.

        Args:
            async_session_manager: Database session manager for persistence
        """
        self.async_session = async_session_manager
        self.default_config = FlowConfig()
        self._pending_batches: Dict[str, List[ProcessedData]] = {}
        self._callbacks: Dict[str, Callable] = {}
        self._stats = {
            'total_routed': 0,
            'database_writes': 0,
            'callback_invocations': 0,
            'errors': 0
        }

        logger.info("Flow Control Hub initialized")

    # ========================================================================
    # Primary Routing Methods
    # ========================================================================

    async def route_data(
        self,
        data: ProcessedData,
        config: Optional[FlowConfig] = None
    ) -> bool:
        """
        Route a single processed data point to its destination.

        Args:
            data: Processed data to route
            config: Optional routing configuration (uses default if None)

        Returns:
            True if routing successful, False otherwise
        """
        cfg = config or self.default_config

        try:
            if cfg.destination == FlowDestination.DATABASE:
                success = await self._route_to_database(data)
            elif cfg.destination == FlowDestination.MEMORY:
                success = self._route_to_memory(data)
            elif cfg.destination == FlowDestination.CALLBACK:
                success = await self._route_to_callback(data, cfg.callback)
            elif cfg.destination == FlowDestination.BROADCAST:
                success = await self._route_to_broadcast(data, cfg)
            else:
                logger.warning(f"Unknown destination: {cfg.destination}")
                success = False

            if success:
                self._stats['total_routed'] += 1
            else:
                self._stats['errors'] += 1

            return success

        except Exception as e:
            logger.exception(f"Error routing data for {data.image_id}: {e}")
            self._stats['errors'] += 1
            return False

    async def route_batch(
        self,
        batch_data: List[ProcessedData],
        config: Optional[FlowConfig] = None
    ) -> int:
        """
        Route multiple processed data points with order preservation.

        Args:
            batch_data: List of processed data to route
            config: Optional routing configuration

        Returns:
            Number of successfully routed items
        """
        cfg = config or self.default_config

        # Ensure order preservation if requested
        if cfg.preserve_order:
            batch_data = sorted(batch_data, key=lambda d: d.sequence_num)

        success_count = 0

        # Route based on destination
        if cfg.destination == FlowDestination.DATABASE:
            success_count = await self._route_batch_to_database(batch_data)
        else:
            # For other destinations, route individually
            for data in batch_data:
                if await self.route_data(data, config):
                    success_count += 1

        self._stats['total_routed'] += success_count
        if success_count < len(batch_data):
            self._stats['errors'] += (len(batch_data) - success_count)

        return success_count

    # ========================================================================
    # Destination-Specific Routing
    # ========================================================================

    async def _route_to_database(self, data: ProcessedData) -> bool:
        """Route single data point to database."""
        if not self.async_session:
            logger.error("No async session manager configured")
            return False

        success = await self.async_session.save_caption(data.image_id, data.content)
        if success:
            self._stats['database_writes'] += 1
            logger.debug(f"Saved caption for {data.image_id} to database")

        return success

    async def _route_batch_to_database(self, batch_data: List[ProcessedData]) -> int:
        """Route batch of data to database efficiently."""
        if not self.async_session:
            logger.error("No async session manager configured")
            return 0

        captions_data = [
            {'image_id': item.image_id, 'caption': item.content}
            for item in batch_data
        ]

        # Log what we're trying to save for debugging
        logger.debug(f"Attempting to save {len(batch_data)} captions: {[item.image_id for item in batch_data][:5]}...")

        success_count = await self.async_session.save_captions_batch(captions_data)
        self._stats['database_writes'] += success_count

        if success_count == len(batch_data):
            logger.info(f"Saved {success_count} captions to database")
        else:
            # Log details about what failed for debugging
            logger.warning(
                f"Only saved {success_count}/{len(batch_data)} captions to database. "
                f"Image IDs attempted: {[item.image_id for item in batch_data]}"
            )

        return success_count

    def _route_to_memory(self, data: ProcessedData) -> bool:
        """Route data to memory (no-op for now, could cache here)."""
        logger.debug(f"Kept data for {data.image_id} in memory only")
        return True

    async def _route_to_callback(
        self,
        data: ProcessedData,
        callback: Optional[Callable]
    ) -> bool:
        """Route data to a custom callback function."""
        if not callback:
            logger.error("No callback provided for CALLBACK destination")
            return False

        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(data)
            else:
                callback(data)

            self._stats['callback_invocations'] += 1
            return True

        except Exception as e:
            logger.exception(f"Callback error for {data.image_id}: {e}")
            return False

    async def _route_to_broadcast(
        self,
        data: ProcessedData,
        config: FlowConfig
    ) -> bool:
        """Route data to multiple destinations."""
        # Broadcast to both database and callback (example)
        results = await asyncio.gather(
            self._route_to_database(data),
            self._route_to_callback(data, config.callback) if config.callback else asyncio.sleep(0),
            return_exceptions=True
        )

        # Consider it successful if at least one destination succeeded
        return any(r is True for r in results if not isinstance(r, Exception))

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def register_callback(self, name: str, callback: Callable):
        """
        Register a named callback for later use.

        Args:
            name: Callback identifier
            callback: Callable that accepts ProcessedData
        """
        self._callbacks[name] = callback
        logger.info(f"Registered callback: {name}")

    def get_callback(self, name: str) -> Optional[Callable]:
        """Get a registered callback by name."""
        return self._callbacks.get(name)

    def get_stats(self) -> Dict[str, int]:
        """Get routing statistics."""
        return self._stats.copy()

    def reset_stats(self):
        """Reset routing statistics."""
        self._stats = {
            'total_routed': 0,
            'database_writes': 0,
            'callback_invocations': 0,
            'errors': 0
        }
        logger.info("Flow control statistics reset")


# ============================================================================
# Convenience Functions
# ============================================================================

def create_processed_data(
    image_id: str,
    content: str,
    model_name: str,
    parameters: Optional[Dict] = None,
    metadata: Optional[Dict] = None,
    sequence_num: int = 0
) -> ProcessedData:
    """
    Convenience function to create ProcessedData instance.

    Args:
        image_id: Image identifier
        content: Processed text (caption, tags, etc.)
        model_name: Name of the model that generated the content
        parameters: Generation parameters used
        metadata: Additional metadata
        sequence_num: Sequence number for ordering

    Returns:
        ProcessedData instance
    """
    return ProcessedData(
        image_id=image_id,
        content=content,
        model_name=model_name,
        parameters=parameters or {},
        metadata=metadata or {},
        sequence_num=sequence_num
    )


def create_flow_config(
    destination: str = "database",
    preserve_order: bool = True,
    batch_size: int = 10,
    callback: Optional[Callable] = None,
    **metadata
) -> FlowConfig:
    """
    Convenience function to create FlowConfig instance.

    Args:
        destination: Destination type ("database", "memory", "callback", "broadcast")
        preserve_order: Whether to preserve data ordering
        batch_size: Batch size for database operations
        callback: Optional callback function
        **metadata: Additional metadata

    Returns:
        FlowConfig instance
    """
    dest_enum = FlowDestination[destination.upper()]

    return FlowConfig(
        destination=dest_enum,
        preserve_order=preserve_order,
        batch_size=batch_size,
        callback=callback,
        metadata=metadata
    )
