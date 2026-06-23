/**
 * useFileUpload hook — handles file upload with drag-and-drop.
 */

import { useState, useCallback } from 'react';
import { uploadFile, uploadResume } from '../services/api';
import useAppStore from '../store/appStore';

const useFileUpload = (type = 'general') => {
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState(null);
  const { addUploadedFile } = useAppStore();

  const upload = useCallback(async (file) => {
    setUploading(true);
    setProgress(0);
    setError(null);

    try {
      // Simulate progress
      const progressInterval = setInterval(() => {
        setProgress((prev) => Math.min(prev + 10, 90));
      }, 200);

      let result;
      if (type === 'resume') {
        result = await uploadResume(file);
      } else {
        result = await uploadFile(file);
      }

      clearInterval(progressInterval);
      setProgress(100);
      addUploadedFile(result);

      setTimeout(() => setProgress(0), 1000);
      return result;
    } catch (err) {
      setError(err.message || 'Upload failed');
      return null;
    } finally {
      setUploading(false);
    }
  }, [type]);

  return { upload, uploading, progress, error };
};

export default useFileUpload;
