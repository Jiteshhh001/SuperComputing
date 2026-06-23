/**
 * Header — top bar with agent indicator, search, and actions.
 */

import React from 'react';
import { HiOutlineSparkles } from 'react-icons/hi2';
import useAppStore from '../../store/appStore';
import './Header.css';

const agentLabels = {
  general: { name: 'General Assistant', color: 'var(--accent-primary)' },
  resume: { name: 'Resume Tailor', color: 'var(--agent-resume)' },
  research: { name: 'Deep Research', color: 'var(--agent-research)' },
  coding: { name: 'Coding Agent', color: 'var(--agent-coding)' },
  interview: { name: 'Interview Coach', color: 'var(--agent-interview)' },
};

const Header = () => {
  const { activeAgent, sidebarOpen } = useAppStore();
  const agent = agentLabels[activeAgent] || agentLabels.general;

  return (
    <header className="app-header" style={{ marginLeft: sidebarOpen ? 'var(--sidebar-width)' : 'var(--sidebar-collapsed)' }}>
      <div className="header-left">
        <div className="agent-indicator" style={{ '--indicator-color': agent.color }}>
          <HiOutlineSparkles size={16} />
          <span>{agent.name}</span>
        </div>
      </div>
      <div className="header-right">
        <div className="header-status">
          <span className="status-dot" />
          <span className="status-text">Online</span>
        </div>
      </div>
    </header>
  );
};

export default Header;
