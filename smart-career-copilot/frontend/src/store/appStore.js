/**
 * Zustand store — global state management for the application.
 */

import { create } from 'zustand';

const useAppStore = create((set, get) => ({
  // ── Active Agent ───────────────────────────────
  activeAgent: 'general',
  setActiveAgent: (agent) => set({ activeAgent: agent }),

  // ── Sessions ───────────────────────────────────
  sessions: [],
  currentSessionId: null,
  setSessions: (sessions) => set({ sessions }),
  setCurrentSessionId: (id) => set({ currentSessionId: id }),
  addSession: (session) =>
    set((state) => ({ sessions: [session, ...state.sessions] })),
  removeSession: (id) =>
    set((state) => ({
      sessions: state.sessions.filter((s) => s.session_id !== id),
      currentSessionId: state.currentSessionId === id ? null : state.currentSessionId,
    })),

  // ── Messages ───────────────────────────────────
  messages: [],
  setMessages: (messages) => set({ messages }),
  addMessage: (message) =>
    set((state) => ({ messages: [...state.messages, message] })),
  clearMessages: () => set({ messages: [] }),

  // ── Loading ────────────────────────────────────
  isLoading: false,
  setIsLoading: (loading) => set({ isLoading: loading }),

  // ── Sidebar ────────────────────────────────────
  sidebarOpen: true,
  toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })),

  // ── Upload ─────────────────────────────────────
  uploadedFiles: [],
  addUploadedFile: (file) =>
    set((state) => ({ uploadedFiles: [...state.uploadedFiles, file] })),
  removeUploadedFile: (id) =>
    set((state) => ({
      uploadedFiles: state.uploadedFiles.filter((f) => f.file_id !== id),
    })),
  clearUploadedFiles: () => set({ uploadedFiles: [] }),

  // ── Resume State ───────────────────────────────
  resumeData: null,
  setResumeData: (data) => set({ resumeData: data }),

  // ── Research State ─────────────────────────────
  researchReport: null,
  setResearchReport: (report) => set({ researchReport: report }),

  // ── Interview State ────────────────────────────
  interviewSession: null,
  setInterviewSession: (session) => set({ interviewSession: session }),

  // ── Thinking Steps ─────────────────────────────
  thinkingSteps: [],
  setThinkingSteps: (steps) => set({ thinkingSteps: steps }),
  addThinkingStep: (step) =>
    set((state) => ({ thinkingSteps: [...state.thinkingSteps, step] })),
  clearThinkingSteps: () => set({ thinkingSteps: [] }),

  // ── Notification ───────────────────────────────
  notification: null,
  setNotification: (notification) => set({ notification }),
  clearNotification: () => set({ notification: null }),
}));

export default useAppStore;
