import api from './api';

export const registerUser = async (name, email, password) => {
  const response = await api.post('/api/auth/register', { name, email, password });
  return response.data; // expects { token, user: { name, email } }
};

export const loginUser = async (email, password) => {
  const response = await api.post('/api/auth/login', { email, password });
  return response.data; // expects { token, user: { name, email } }
};