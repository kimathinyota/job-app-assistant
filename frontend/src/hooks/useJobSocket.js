// frontend/src/hooks/useJobSocket.js
import { useEffect, useRef } from 'react';

// Use a singleton for the connection to avoid reconnects on re-renders
let socket = null;
let activeSubscribers = 0;

export const useJobSocket = (userId, onMessage) => {
    const messageHandlerRef = useRef(onMessage);

    // Keep the handler current without re-triggering effect
    useEffect(() => {
        messageHandlerRef.current = onMessage;
    }, [onMessage]);

    useEffect(() => {
        if (!userId) return;

        // 1. Connect if not connected
        if (!socket || socket.readyState !== WebSocket.OPEN) {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const host = window.location.hostname === 'localhost' ? 'localhost:8000' : window.location.host;
            
            socket = new WebSocket(`${protocol}//${host}/ws/${userId}`);
            
            socket.onopen = () => console.log("🟢 WS Connected");
            socket.onclose = () => console.log("🔴 WS Disconnected");
        }

        // 2. Add Listener
        const handleEvent = (event) => {
            try {
                const data = JSON.parse(event.data);
                if (messageHandlerRef.current) {
                    messageHandlerRef.current(data);
                }
            } catch (e) {
                console.error("WS Parse Error", e);
            }
        };

        socket.addEventListener('message', handleEvent);
        activeSubscribers++;

        // 3. Cleanup
        return () => {
            socket.removeEventListener('message', handleEvent);
            activeSubscribers--;
            // Optional: Close socket if no components are listening
            // if (activeSubscribers === 0 && socket) { socket.close(); socket = null; }
        };
    }, [userId]);
};