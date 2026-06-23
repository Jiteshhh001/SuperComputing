/**
 * useChat hook — handles sending messages and managing chat state.
 */

import { useCallback } from 'react';
import { sendMessage, getSessionMessages } from '../services/api';
import useAppStore from '../store/appStore';

const useChat = () => {
  const {
    messages, addMessage, setMessages, isLoading, setIsLoading,
    currentSessionId, setCurrentSessionId, activeAgent,
    setThinkingSteps, clearThinkingSteps, addSession,
  } = useAppStore();

  const send = useCallback(async (text, agentType = null, attachments = [], context = {}) => {
    if ((!text.trim() && (!attachments || attachments.length === 0)) || isLoading) return;

    // Add user message to UI immediately
    const userMsg = {
      id: Date.now().toString(),
      role: 'user',
      content: text,
      metadata: { attachments },
      timestamp: new Date().toISOString(),
    };
    addMessage(userMsg);
    setIsLoading(true);
    clearThinkingSteps();

    try {
      const response = await sendMessage(
        text,
        currentSessionId,
        agentType || activeAgent,
        context,
        attachments
      );

      // Update session ID if new
      if (!currentSessionId && response.session_id) {
        setCurrentSessionId(response.session_id);
        addSession({
          session_id: response.session_id,
          agent_type: response.agent_used,
          title: text.substring(0, 80),
          created_at: new Date().toISOString(),
        });
      }

      // Add assistant message
      const assistantMsg = {
        id: response.message?.id || (Date.now() + 1).toString(),
        role: 'assistant',
        content: response.message?.content || 'No response received.',
        agent_type: response.agent_used,
        sources: response.sources || [],
        artifacts: response.artifacts || {},
        timestamp: new Date().toISOString(),
      };
      addMessage(assistantMsg);

      // Set thinking steps
      if (response.thinking_steps?.length) {
        setThinkingSteps(response.thinking_steps);
      }

    } catch (error) {
      console.error('Chat error:', error);
      addMessage({
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: `⚠️ Error: ${
          error.response?.data?.detail 
            ? (typeof error.response.data.detail === 'string' 
                ? error.response.data.detail 
                : JSON.stringify(error.response.data.detail))
            : (error.message || 'Failed to connect to the server. Please ensure the backend is running.')
        }`,
        timestamp: new Date().toISOString(),
      });
    } finally {
      setIsLoading(false);
    }
  }, [currentSessionId, activeAgent, isLoading]);

  const loadSession = useCallback(async (sessionId) => {
    try {
      const data = await getSessionMessages(sessionId);
      setMessages(data.messages || []);
      setCurrentSessionId(sessionId);
    } catch (error) {
      console.error('Load session error:', error);
    }
  }, []);

  return { send, loadSession, messages, isLoading };
};

export default useChat;
