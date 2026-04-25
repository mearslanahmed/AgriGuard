// Central axios instance. Need to only change the base URL here when deploying.

import axios from 'axios';

// During development: use your PC's local IP (not localhost — phone can't reach it)
// Example: 'http://192.168.1.5:5000'
import { BACKEND_URL } from '../config';
const BASE_URL = BACKEND_URL;

const api = axios.create({
  baseURL: BASE_URL,
  timeout: 15000, // 15 seconds — image uploads can be slow
  headers: {
    'Content-Type': 'application/json',
  },
});

// Attach JWT token to every request automatically
api.interceptors.request.use((config) => {
  // Token will be injected per-request from SecureStore when needed
  return config;
});

export default api;