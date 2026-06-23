/**
 * App — root component wrapping the router.
 */

import React from 'react';
import { RouterProvider } from 'react-router-dom';
import router from './router';

import './styles/index.css';
import './styles/animations.css';
import './styles/glassmorphism.css';

function App() {
  return <RouterProvider router={router} />;
}

export default App;
