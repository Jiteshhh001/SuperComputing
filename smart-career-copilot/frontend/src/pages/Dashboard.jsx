/**
 * Dashboard — landing page with agent cards and quick actions.
 */

import React from 'react';
import { useNavigate } from 'react-router-dom';
import {
  HiOutlineDocumentText,
  HiOutlineMagnifyingGlass,
  HiOutlineCodeBracket,
  HiOutlineAcademicCap,
} from 'react-icons/hi2';
import AnimatedCard from '../components/common/AnimatedCard';
import ChatWindow from '../components/chat/ChatWindow';
import ChatInput from '../components/chat/ChatInput';
import useChat from '../hooks/useChat';
import useAppStore from '../store/appStore';
import './Dashboard.css';

const agentCards = [
  {
    id: 'resume',
    title: 'Resume Tailor',
    description: 'Upload your resume, get ATS scores, skill gap analysis, and AI-improved bullets.',
    icon: <HiOutlineDocumentText size={32} />,
    color: 'var(--agent-resume)',
    path: '/resume',
  },
  {
    id: 'research',
    title: 'Deep Research',
    description: 'Search the web, academic papers, and generate comprehensive research reports.',
    icon: <HiOutlineMagnifyingGlass size={32} />,
    color: 'var(--agent-research)',
    path: '/research',
  },
  {
    id: 'coding',
    title: 'Coding Agent',
    description: 'Describe a task and let AI plan, code, test, review, and iterate automatically.',
    icon: <HiOutlineCodeBracket size={32} />,
    color: 'var(--agent-coding)',
    path: '/coding',
  },
  {
    id: 'interview',
    title: 'Interview Coach',
    description: 'Practice technical and behavioral interviews with real-time AI feedback.',
    icon: <HiOutlineAcademicCap size={32} />,
    color: 'var(--agent-interview)',
    path: '/interview',
  },
];

const Dashboard = () => {
  const navigate = useNavigate();
  const { send, messages, isLoading } = useChat();
  const { setActiveAgent } = useAppStore();

  const handleAgentClick = (agent) => {
    setActiveAgent(agent.id);
    navigate(agent.path);
  };

  return (
    <div className="dashboard">
      {messages.length === 0 ? (
        <div className="dashboard-welcome animate-fade-in">
          <div className="welcome-header">
            <div className="welcome-icon animate-float">✦</div>
            <h1>Smart Career Copilot</h1>
            <p className="welcome-subtitle">
              Your AI-powered career development platform. Choose an agent below or start chatting.
            </p>
          </div>

          <div className="agent-grid stagger-children">
            {agentCards.map((agent) => (
              <AnimatedCard
                key={agent.id}
                icon={agent.icon}
                title={agent.title}
                description={agent.description}
                color={agent.color}
                onClick={() => handleAgentClick(agent)}
              />
            ))}
          </div>

          <div className="quick-stats glass-panel">
            <div className="stat-item">
              <span className="stat-value">4</span>
              <span className="stat-label">AI Agents</span>
            </div>
            <div className="stat-divider" />
            <div className="stat-item">
              <span className="stat-value">15+</span>
              <span className="stat-label">Tools</span>
            </div>
            <div className="stat-divider" />
            <div className="stat-item">
              <span className="stat-value">RAG</span>
              <span className="stat-label">Powered</span>
            </div>
            <div className="stat-divider" />
            <div className="stat-item">
              <span className="stat-value">∞</span>
              <span className="stat-label">Memory</span>
            </div>
          </div>
        </div>
      ) : (
        <ChatWindow messages={messages} isLoading={isLoading} />
      )}

      <ChatInput onSend={(text, attachments) => send(text, null, attachments)} isLoading={isLoading} />
    </div>
  );
};

export default Dashboard;
