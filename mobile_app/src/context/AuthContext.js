// manages login state across the whole app.

import React, { createContext, useState, useEffect, useContext } from 'react';
import * as SecureStore from 'expo-secure-store';

const AuthContext = createContext();

export function AuthProvider({ children }) {
  const [userToken, setUserToken] = useState(null);
  const [userInfo, setUserInfo] = useState(null);
  const [isLoading, setIsLoading] = useState(true); // checking stored token on startup

  // On app launch, check if a token is already saved
  useEffect(() => {
    const loadToken = async () => {
      try {
        const token = await SecureStore.getItemAsync('userToken');
        const info = await SecureStore.getItemAsync('userInfo');
        if (token) {
          setUserToken(token);
          setUserInfo(JSON.parse(info));
        }
      } catch (e) {
        console.log('Failed to load token', e);
      } finally {
        setIsLoading(false);
      }
    };
    loadToken();
  }, []);

  const login = async (token, info) => {
    await SecureStore.setItemAsync('userToken', token);
    await SecureStore.setItemAsync('userInfo', JSON.stringify(info));
    setUserToken(token);
    setUserInfo(info);
  };

  const logout = async () => {
    await SecureStore.deleteItemAsync('userToken');
    await SecureStore.deleteItemAsync('userInfo');
    setUserToken(null);
    setUserInfo(null);
  };
  const updateUserInfo = async (updated) => {
  await SecureStore.setItemAsync('userInfo', JSON.stringify(updated));
  setUserInfo(updated);
  };

  return (
    <AuthContext.Provider value={{ userToken, userInfo, isLoading, login, logout, updateUserInfo }}>
      {children}
    </AuthContext.Provider>
  );
}

// Custom hook — use this in any screen instead of importing AuthContext directly
export function useAuth() {
  return useContext(AuthContext);
}