/**
 * Router configuration — maps routes to pages.
 */

import React from 'react';
import { createBrowserRouter } from 'react-router-dom';
import MainLayout from './components/layout/MainLayout';
import Dashboard from './pages/Dashboard';
import ResumeAgent from './pages/ResumeAgent';
import ResearchAgent from './pages/ResearchAgent';
import CodingAgent from './pages/CodingAgent';
import InterviewAgent from './pages/InterviewAgent';

const router = createBrowserRouter([
  {
    path: '/',
    element: <MainLayout />,
    children: [
      { index: true, element: <Dashboard /> },
      { path: 'resume', element: <ResumeAgent /> },
      { path: 'research', element: <ResearchAgent /> },
      { path: 'coding', element: <CodingAgent /> },
      { path: 'interview', element: <InterviewAgent /> },
    ],
  },
]);

export default router;
