/**
 * API service — Axios client for backend communication.
 */

import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: `${API_BASE}/api`,
  headers: { 'Content-Type': 'application/json' },
  timeout: 120000,
});

// ── Chat ────────────────────────────────────────

export const sendMessage = async (message, sessionId = null, agentType = null, context = {}, attachments = []) => {
  const { data } = await api.post('/chat/send', {
    message,
    session_id: sessionId,
    agent_type: agentType,
    context,
    attachments,
  });
  return data;
};

export const getSessions = async () => {
  const { data } = await api.get('/chat/sessions');
  return data;
};

export const getSessionMessages = async (sessionId) => {
  const { data } = await api.get(`/chat/sessions/${sessionId}/messages`);
  return data;
};

export const deleteSession = async (sessionId) => {
  const { data } = await api.delete(`/chat/sessions/${sessionId}`);
  return data;
};

// ── Resume ──────────────────────────────────────

export const uploadResume = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  const { data } = await api.post('/resume/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return data;
};

export const analyzeResume = async (fileId, resumeText, jobDescription = '') => {
  const formData = new FormData();
  formData.append('file_id', fileId);
  formData.append('resume_text', resumeText);
  if (jobDescription) formData.append('job_description', jobDescription);
  const { data } = await api.post('/resume/analyze', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return data;
};

export const generateCoverLetter = async (resumeText, jobDescription, companyName) => {
  const formData = new FormData();
  formData.append('resume_text', resumeText);
  formData.append('job_description', jobDescription);
  formData.append('company_name', companyName);
  const { data } = await api.post('/resume/cover-letter', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return data;
};

// ── Research ────────────────────────────────────

export const researchQuery = async (query, depth = 'standard', sources = ['web', 'arxiv']) => {
  const { data } = await api.post('/research/query', {
    query,
    depth,
    sources,
    max_results: 5,
  });
  return data;
};

// ── Coding ──────────────────────────────────────

export const executeCodingTask = async (taskDescription, language = 'python', requirements = []) => {
  const { data } = await api.post('/coding/execute', {
    task_description: taskDescription,
    language,
    requirements,
  });
  return data;
};

export const getWorkspaceFiles = async () => {
  const { data } = await api.get('/coding/files');
  return data;
};

// ── Interview ───────────────────────────────────

export const startInterview = async (config) => {
  const { data } = await api.post('/interview/start', config);
  return data;
};

export const submitAnswer = async (sessionId, questionId, answer) => {
  const { data } = await api.post(`/interview/answer?session_id=${sessionId}&question_id=${questionId}&answer=${encodeURIComponent(answer)}`);
  return data;
};

export const getNextQuestion = async (sessionId) => {
  const { data } = await api.get(`/interview/question/${sessionId}`);
  return data;
};

export const getScorecard = async (sessionId) => {
  const { data } = await api.get(`/interview/scorecard/${sessionId}`);
  return data;
};

// ── Files ───────────────────────────────────────

export const uploadFile = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  const { data } = await api.post('/files/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return data;
};

// ── Health ──────────────────────────────────────

export const healthCheck = async () => {
  const { data } = await api.get('/health');
  return data;
};

// ── WebSocket ───────────────────────────────────

export const createWebSocket = (sessionId) => {
  const wsBase = API_BASE.replace('http', 'ws');
  return new WebSocket(`${wsBase}/api/chat/ws/${sessionId}`);
};

export default api;
