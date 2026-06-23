/**
 * MainLayout — root layout with sidebar and content area.
 */

import React from 'react';
import { Outlet } from 'react-router-dom';
import Sidebar from './Sidebar';
import Header from './Header';
import useAppStore from '../../store/appStore';
import './MainLayout.css';

const MainLayout = () => {
  const { sidebarOpen } = useAppStore();

  return (
    <div className="main-layout">
      <Sidebar />
      <div
        className="main-content"
        style={{
          marginLeft: sidebarOpen ? 'var(--sidebar-width)' : 'var(--sidebar-collapsed)',
        }}
      >
        <Header />
        <main className="content-area">
          <Outlet />
        </main>
      </div>
    </div>
  );
};

export default MainLayout;
