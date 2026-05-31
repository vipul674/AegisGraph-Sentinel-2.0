import pytest
import time

from src.api.websocket_manager import WebSocketManager


@pytest.mark.anyio
async def test_websocket_connect_and_disconnect():
    manager = WebSocketManager()
    ws = MockWebSocket()
    
    accepted = await manager.connect(ws, "client_1")
    assert accepted is True
    assert ws.accepted is True
    assert "client_1" in manager.active_connections
    
    await manager.disconnect("client_1")
    assert "client_1" not in manager.active_connections

@pytest.mark.anyio
async def test_reconnect_backoff():
    manager = WebSocketManager(max_reconnect_attempts=3)
    ws = MockWebSocket()
    
    for i in range(3):
        assert await manager.connect(ws, "flood_client") is True
        await manager.disconnect("flood_client")
        
    ws_rejected = MockWebSocket()
    accepted = await manager.connect(ws_rejected, "flood_client")
    assert accepted is False
    assert ws_rejected.closed is True
    assert ws_rejected.close_code == 1008

@pytest.mark.anyio
async def test_heartbeat_and_stale_cleanup():
    manager = WebSocketManager(heartbeat_timeout=0.1)
    ws1 = MockWebSocket()
    ws2 = MockWebSocket()
    
    await manager.connect(ws1, "active_client")
    await manager.connect(ws2, "stale_client")
    
    await asyncio.sleep(0.15)
    
    await manager.heartbeat("active_client")
    
    await manager.cleanup_stale_connections()
    
    assert "active_client" in manager.active_connections
    assert "stale_client" not in manager.active_connections
    assert ws2.closed is True

@pytest.mark.anyio
async def test_broadcast():
    manager = WebSocketManager()
    ws1 = MockWebSocket()
    ws2 = MockWebSocket()
    
    await manager.connect(ws1, "client_1")
    await manager.connect(ws2, "client_2")
    
    await manager.broadcast({"fraud_alert": "high"})
    
    assert len(ws1.messages) == 1
    assert ws1.messages[0] == {"fraud_alert": "high"}
    assert len(ws2.messages) == 1
    assert ws2.messages[0] == {"fraud_alert": "high"}
@pytest.mark.asyncio
async def test_disconnect_history_eviction_drops_stale_clients():
    manager = WebSocketManager(disconnect_history_ttl=1.0, max_disconnect_history_entries=10)
    now = time.time()
    manager.disconnect_history = {
        "stale-client": [now - 10.0, now - 9.0],
        "fresh-client": [now - 0.2],
    }

    await manager.evict_stale_disconnect_history()

    assert "stale-client" not in manager.disconnect_history
    assert "fresh-client" in manager.disconnect_history


@pytest.mark.asyncio
async def test_disconnect_history_eviction_caps_total_clients():
    manager = WebSocketManager(disconnect_history_ttl=60.0, max_disconnect_history_entries=2)
    now = time.time()
    manager.disconnect_history = {
        "client-a": [now - 30.0],
        "client-b": [now - 20.0],
        "client-c": [now - 10.0],
    }

    await manager.evict_stale_disconnect_history()

    assert len(manager.disconnect_history) == 2
    assert "client-a" not in manager.disconnect_history
