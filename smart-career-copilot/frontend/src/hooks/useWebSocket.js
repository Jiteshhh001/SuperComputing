/**
 * useWebSocket hook — manages WebSocket connection for real-time streaming.
 */

import { useEffect, useRef, useState, useCallback } from 'react';
import { createWebSocket } from '../services/api';

const useWebSocket = (sessionId) => {
  const wsRef = useRef(null);
  const [connected, setConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState(null);

  const connect = useCallback(() => {
    if (!sessionId || wsRef.current) return;

    const ws = createWebSocket(sessionId);

    ws.onopen = () => {
      setConnected(true);
      console.log('WebSocket connected:', sessionId);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        setLastMessage(data);
      } catch (e) {
        setLastMessage({ type: 'token', content: event.data });
      }
    };

    ws.onclose = () => {
      setConnected(false);
      wsRef.current = null;
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setConnected(false);
    };

    wsRef.current = ws;
  }, [sessionId]);

  const disconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
      setConnected(false);
    }
  }, []);

  const sendWsMessage = useCallback((message) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
    }
  }, []);

  useEffect(() => {
    return () => disconnect();
  }, [disconnect]);

  return { connect, disconnect, sendWsMessage, connected, lastMessage };
};

export default useWebSocket;
