import axios from 'axios';

const configuredApiUrl = import.meta.env.VITE_API_URL;

if (import.meta.env.PROD && !configuredApiUrl) {
  throw new Error('Missing VITE_API_URL for production build/runtime.');
}

if (import.meta.env.PROD && configuredApiUrl?.startsWith('http://')) {
  throw new Error('VITE_API_URL must use https:// in production.');
}

const API_URL = configuredApiUrl || 'http://localhost:8000';
let authTokenProvider = null;

const client = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const setAuthTokenProvider = (provider) => {
  authTokenProvider = provider;
};

// Optional bearer token support (for Clerk-integrated deployments)
client.interceptors.request.use(
  async (config) => {
    const token = typeof authTokenProvider === 'function' ? await authTokenProvider() : null;

    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

export const authApi = {
  me: async () => {
    const response = await client.get('/auth/me');
    return response.data;
  },
};

export const scoringApi = {
  mergePlan: async (files) => {
    const formData = new FormData();
    files.forEach((f) => formData.append('files', f));

    const response = await client.post('/merge-plan', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;
  },

  // ━━ SYNC TRAINING (legacy, still works) ━━
  train: async (modelName, files, targetCol = null, mode = "supervised") => {
    const formData = new FormData();
    files.forEach((f) => formData.append('files', f));

    let url = `/train?model_name=${encodeURIComponent(modelName)}&mode=${encodeURIComponent(mode)}`;
    if (targetCol) url += `&target_column=${encodeURIComponent(targetCol)}`;

    const response = await client.post(url, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;
  },

  // ━━ ASYNC TRAINING (NEW - Production Ready) ━━
  /**
   * Start background training job. Returns immediately with job_id.
   * @param {string} modelName - Model name
   * @param {File[]} files - CSV files to train on
   * @param {string} targetCol - Binary target column name (optional for auto-detect)
   * @param {string} mode - 'supervised' (default) or 'unsupervised'
   * @returns {Promise} {job_id, status, message, poll_url, result_url}
   */
  trainAsync: async (modelName, files, targetCol = null, mode = "supervised") => {
    const formData = new FormData();
    files.forEach((f) => formData.append('files', f));

    let url = `/train/async?model_name=${encodeURIComponent(modelName)}&mode=${encodeURIComponent(mode)}`;
    if (targetCol) url += `&target_column=${encodeURIComponent(targetCol)}`;

    const response = await client.post(url, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;
  },

  /**
   * Poll for training job status. Call this every 3-5 seconds.
   * @param {string} jobId - Job ID from trainAsync
   * @returns {Promise} Job status, progress (0-100), current_step, etc.
   */
  getTrainingStatus: async (jobId) => {
    const response = await client.get(`/train/status/${encodeURIComponent(jobId)}`);
    return response.data;
  },

  /**
   * Get final training results (call when status = 'completed').
   * @param {string} jobId - Job ID from trainAsync
   * @returns {Promise} Final metrics, model info, results
   */
  getTrainingResult: async (jobId) => {
    const response = await client.get(`/train/${encodeURIComponent(jobId)}/result`);
    return response.data;
  },

  /**
   * List all training jobs for current user.
   * @param {number} limit - Max jobs to return (default 50)
   * @returns {Promise} List of jobs
   */
  listTrainingJobs: async (limit = 50) => {
    const response = await client.get(`/train/jobs?limit=${limit}`);
    return response.data;
  },

  score: async (modelName, files, autoSelectModel = false) => {
    const formData = new FormData();
    files.forEach((f) => formData.append('files', f));

    const url =
      `/score-csv?model_name=${encodeURIComponent(modelName)}&auto_select_model=${encodeURIComponent(autoSelectModel)}`;
    const response = await client.post(url, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;
  },

  feedback: async (modelName, file, outcomeColumn = null, autoRetrain = false, feedbackWeight = 2) => {
    const formData = new FormData();
    formData.append('file', file);

    let url = `/feedback?model_name=${encodeURIComponent(modelName)}`;
    if (outcomeColumn) url += `&outcome_column=${encodeURIComponent(outcomeColumn)}`;
    url += `&auto_retrain=${encodeURIComponent(autoRetrain)}`;
    url += `&feedback_weight=${encodeURIComponent(feedbackWeight)}`;

    const response = await client.post(url, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;
  },

  retrainFromFeedback: async (modelName, feedbackWeight = 2) => {
    const url = `/retrain-from-feedback?model_name=${encodeURIComponent(modelName)}&feedback_weight=${encodeURIComponent(feedbackWeight)}`;
    const response = await client.post(url);
    return response.data;
  },
  retrainSegmentFromFeedback: async (modelName, segmentDimension, segmentValue, feedbackWeight = 2) => {
    const url = `/retrain-segment-feedback?model_name=${encodeURIComponent(modelName)}&segment_dimension=${encodeURIComponent(segmentDimension)}&segment_value=${encodeURIComponent(segmentValue)}&feedback_weight=${encodeURIComponent(feedbackWeight)}`;
    const response = await client.post(url);
    return response.data;
  },
};

export const modelsApi = {
  list: async () => {
    const response = await client.get('/models');
    return response.data;
  },
  get: async (modelName) => {
    const response = await client.get(`/models/${encodeURIComponent(modelName)}`);
    return response.data;
  },
  delete: async (modelName) => {
    const response = await client.delete(`/models/${encodeURIComponent(modelName)}`);
    return response.data;
  },
};

export default client;
