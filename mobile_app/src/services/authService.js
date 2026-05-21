import api from './api';

export const sendOtpToEmail = async (email, purpose) => {
  const response = await api.post('/api/auth/send-otp', { email, purpose });
  return response.data; 
};

export const registerUser = async (name, email, password, otp) => {
  const response = await api.post('/api/auth/register', { name, email, password, otp });
  return response.data; 
};

export const executePasswordReset = async (email, otp, newPassword) => {
  const response = await api.post('/api/auth/reset-password', { email, otp, newPassword });
  return response.data; 
};

export const loginUser = async (email, password) => {
  const response = await api.post('/api/auth/login', { email, password });
  return response.data; 
};