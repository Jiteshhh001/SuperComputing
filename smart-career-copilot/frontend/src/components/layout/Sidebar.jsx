/**
 * Sidebar — Agent navigation with session list, inspired by ChatGPT/Claude.
 */

import React from 'react';
import { useNavigate } from 'react-router-dom';
import {
  HiOutlineChatBubbleLeftRight,
  HiOutlineDocumentText,
  HiOutlineMagnifyingGlass,
  HiOutlineCodeBracket,
  HiOutlineAcademicCap,
  HiOutlinePlusCircle,
  HiOutlineTrash,
  HiOutlineChevronLeft,
  HiOutlineChevronRight,
} from 'react-icons/hi2';
import useAppStore from '../../store/appStore';
import useChat from '../../hooks/useChat';
import './Sidebar.css';

const agents = [
  { id: 'general', label: 'General', icon: HiOutlineChatBubbleLeftRight, color: 'var(--accent-primary)' },
  { id: 'resume', label: 'Resume Tailor', icon: HiOutlineDocumentText, color: 'var(--agent-resume)' },
  { id: 'research', label: 'Deep Research', icon: HiOutlineMagnifyingGlass, color: 'var(--agent-research)' },
  { id: 'coding', label: 'Coding Agent', icon: HiOutlineCodeBracket, color: 'var(--agent-coding)' },
  { id: 'interview', label: 'Interview Coach', icon: HiOutlineAcademicCap, color: 'var(--agent-interview)' },
];

const Sidebar = () => {
  const navigate = useNavigate();
  const { loadSession } = useChat();

  const {
    activeAgent, setActiveAgent,
    sessions, currentSessionId, setCurrentSessionId,
    removeSession, sidebarOpen, toggleSidebar,
    clearMessages, setMessages,
  } = useAppStore();

  const handleMobileNav = () => {
    if (window.innerWidth <= 768 && sidebarOpen) {
      toggleSidebar();
    }
  };

  const handleNewChat = () => {
    setCurrentSessionId(null);
    clearMessages();
    navigate(activeAgent === 'general' ? '/' : `/${activeAgent}`);
    handleMobileNav();
  };

  const handleSelectSession = async (session) => {
    setActiveAgent(session.agent_type);
    await loadSession(session.session_id);
    navigate(session.agent_type === 'general' ? '/' : `/${session.agent_type}`);
    handleMobileNav();
  };

  return (
    <aside className={`sidebar glass-sidebar ${sidebarOpen ? 'open' : 'collapsed'}`}>
      {/* Header */}
      <div className="sidebar-header">
        {sidebarOpen && (
          <div className="sidebar-brand animate-fade-in">
            <div className="brand-icon">✦</div>
            <span className="brand-text">Career Copilot</span>
          </div>
        )}
        <button className="sidebar-toggle" onClick={toggleSidebar} title="Toggle sidebar">
          {sidebarOpen ? <HiOutlineChevronLeft size={18} /> : <HiOutlineChevronRight size={18} />}
        </button>
      </div>

      {/* New Chat */}
      <button className="new-chat-btn glass-button" onClick={handleNewChat}>
        <HiOutlinePlusCircle size={20} />
        {sidebarOpen && <span>New Chat</span>}
      </button>

      {/* Agent Navigation */}
      <div className="sidebar-section">
        {sidebarOpen && <div className="section-label">Agents</div>}
        <nav className="agent-nav stagger-children">
          {agents.map((agent) => {
            const Icon = agent.icon;
            const isActive = activeAgent === agent.id;
            return (
              <button
                key={agent.id}
                className={`agent-btn animate-fade-in-up ${isActive ? 'active' : ''}`}
                onClick={() => {
                  setActiveAgent(agent.id);
                  handleNewChat();
                  navigate(agent.id === 'general' ? '/' : `/${agent.id}`);
                }}
                title={agent.label}
                style={{ '--agent-color': agent.color }}
              >
                <Icon size={20} />
                {sidebarOpen && <span>{agent.label}</span>}
                {isActive && <div className="active-indicator" />}
              </button>
            );
          })}
        </nav>
      </div>

      {/* Session History */}
      {sidebarOpen && sessions.length > 0 && (
        <div className="sidebar-section session-section">
          <div className="section-label">Recent Chats</div>
          <div className="session-list">
            {sessions.slice(0, 20).map((session) => (
              <div
                key={session.session_id}
                className={`session-item ${
                  currentSessionId === session.session_id ? 'active' : ''
                }`}
                onClick={() => handleSelectSession(session)}
              >
                <span className="session-title truncate">
                  {session.title || 'New Conversation'}
                </span>
                <button
                  className="session-delete"
                  onClick={(e) => {
                    e.stopPropagation();
                    removeSession(session.session_id);
                  }}
                >
                  <HiOutlineTrash size={14} />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Footer */}
      {sidebarOpen && (
        <div className="sidebar-footer">
          <div className="sidebar-version">v1.0.0 • AI Powered</div>
        </div>
      )}
    </aside>
  );
};

export default Sidebar;
