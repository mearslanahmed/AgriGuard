import React, { useState, useRef } from 'react';
import {
  View, Text, TextInput, TouchableOpacity, StyleSheet,
  ActivityIndicator, KeyboardAvoidingView, Platform, ScrollView, ToastAndroid, Alert
} from 'react-native';
import { useAuth } from '../context/AuthContext';
import { sendOtpToEmail, registerUser } from '../services/authService';
import { Ionicons } from '@expo/vector-icons';

export default function RegisterScreen({ navigation }) {
  const { login } = useAuth();
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [otp, setOtp] = useState('');
  
  const [showPassword, setShowPassword] = useState(false);
  const [isOtpSent, setIsOtpSent] = useState(false);
  const [loading, setLoading] = useState(false);
  const [focusedField, setFocusedField] = useState(null);

  const emailRef = useRef(null);
  const passwordRef = useRef(null);
  const otpRef = useRef(null);

  // Modern Toast alert wrapper
  const showNotification = (msg) => {
    if (Platform.OS === 'android') {
      ToastAndroid.showWithGravityAndOffset(msg, ToastAndroid.LONG, ToastAndroid.BOTTOM, 0, 50);
    } else {
      Alert.alert('AgriGuard', msg);
    }
  };

  const handleRequestOTP = async () => {
    if (!name.trim() || !email.trim() || !password.trim()) {
      showNotification('Please fill in all layout profile rows.');
      return;
    }
    if (password.length < 6) {
      showNotification('Password must be at least 6 characters long.');
      return;
    }

    setLoading(true);
    try {
      const res = await sendOtpToEmail(email.trim(), 'register');
      showNotification('Verification code sent! Please check your email inbox.');
      setIsOtpSent(true);
      setTimeout(() => otpRef.current?.focus(), 150);
    } catch (err) {
      const message = err.response?.data?.message || 'Failed to dispatch verification code.';
      showNotification(message);
    } finally {
      setLoading(false);
    }
  };

  const handleFinalizeRegistration = async () => {
    if (!otp.trim() || otp.length !== 6) {
      showNotification('Please enter the complete 6-digit verification code.');
      return;
    }

    setLoading(true);
    try {
      const data = await registerUser(name.trim(), email.trim(), password, otp.trim());
      showNotification('Account verified successfully! Welcome to AgriGuard.');
      await login(data.token, data.user); 
    } catch (err) {
      const message = err.response?.data?.message || 'Registration dropped. Invalid code.';
      showNotification(message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <KeyboardAvoidingView
      style={styles.container}
      behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
      keyboardVerticalOffset={Platform.OS === 'ios' ? 64 : 0}
    >
      <ScrollView contentContainerStyle={styles.scrollContainer} showsVerticalScrollIndicator={false} keyboardShouldPersistTaps="handled">
        
        <View style={styles.headerSection}>
          <View style={styles.logoBadge}>
            <Ionicons name="leaf" size={32} color="#2e7d32" />
          </View>
          <Text style={styles.title}>AgriGuard</Text>
          <Text style={styles.subtitle}>Verified Farmer Registration Platform</Text>
        </View>

        <View style={styles.cardContainer}>
          <Text style={styles.cardHeaderTitle}>Create Account</Text>

          {/* Name Field */}
          <View style={[styles.inputWrapper, focusedField === 'name' && styles.inputWrapperFocused, isOtpSent && styles.inputDisabled]}>
            <Ionicons name="person-outline" size={20} color={focusedField === 'name' ? '#2e7d32' : '#888'} style={styles.inputIcon} />
            <TextInput
              style={styles.input}
              placeholder="Full Name"
              placeholderTextColor="#a0a0a0"
              autoCapitalize="words"
              value={name}
              onChangeText={setName}
              editable={!isOtpSent}
              onFocus={() => setFocusedField('name')}
              onBlur={() => setFocusedField(null)}
              returnKeyType="next"
              onSubmitEditing={() => emailRef.current?.focus()}
              blurOnSubmit={false}
            />
          </View>

          {/* Email Field */}
          <View style={[styles.inputWrapper, focusedField === 'email' && styles.inputWrapperFocused, isOtpSent && styles.inputDisabled]}>
            <Ionicons name="mail-outline" size={20} color={focusedField === 'email' ? '#2e7d32' : '#888'} style={styles.inputIcon} />
            <TextInput
              ref={emailRef}
              style={styles.input}
              placeholder="Email Address"
              placeholderTextColor="#a0a0a0"
              keyboardType="email-address"
              autoCapitalize="none"
              value={email}
              onChangeText={setEmail}
              editable={!isOtpSent}
              onFocus={() => setFocusedField('email')}
              onBlur={() => setFocusedField(null)}
              returnKeyType="next"
              onSubmitEditing={() => passwordRef.current?.focus()}
              blurOnSubmit={false}
            />
          </View>

          {/* Password Field with Restored View Option */}
          <View style={[styles.inputWrapper, focusedField === 'password' && styles.inputWrapperFocused, isOtpSent && styles.inputDisabled]}>
            <Ionicons name="lock-closed-outline" size={20} color={focusedField === 'password' ? '#2e7d32' : '#888'} style={styles.inputIcon} />
            <TextInput
              ref={passwordRef}
              style={styles.input}
              placeholder="Password (min 6 characters)"
              placeholderTextColor="#a0a0a0"
              secureTextEntry={!showPassword}
              value={password}
              onChangeText={setPassword}
              editable={!isOtpSent}
              onFocus={() => setFocusedField('password')}
              onBlur={() => setFocusedField(null)}
              returnKeyType={isOtpSent ? "next" : "done"}
              onSubmitEditing={() => isOtpSent ? otpRef.current?.focus() : handleRequestOTP()}
              blurOnSubmit={isOtpSent ? false : true}
            />
            {!isOtpSent && (
              <TouchableOpacity onPress={() => setShowPassword(!showPassword)} style={styles.eyeIcon}>
                <Ionicons name={showPassword ? "eye-off-outline" : "eye-outline"} size={20} color="#888" />
              </TouchableOpacity>
            )}
          </View>

          {/* OTP Input Section */}
          {isOtpSent && (
            <View>
              <Text style={styles.otpNoticeText}>Enter the 6-digit confirmation code from your email:</Text>
              <View style={[styles.inputWrapper, focusedField === 'otp' && styles.inputWrapperFocused]}>
                <Ionicons name="shield-checkmark-outline" size={20} color="#2e7d32" style={styles.inputIcon} />
                <TextInput
                  ref={otpRef}
                  style={[styles.input, styles.otpInputText]}
                  placeholder="000000"
                  placeholderTextColor="#a0a0a0"
                  keyboardType="number-pad"
                  maxLength={6}
                  value={otp}
                  onChangeText={setOtp}
                  onFocus={() => setFocusedField('otp')}
                  onBlur={() => setFocusedField(null)}
                  returnKeyType="done"
                  onSubmitEditing={handleFinalizeRegistration}
                />
              </View>
            </View>
          )}

          <TouchableOpacity style={styles.button} onPress={isOtpSent ? handleFinalizeRegistration : handleRequestOTP} disabled={loading}>
            {loading ? (
              <ActivityIndicator color="#fff" />
            ) : (
              <View style={styles.buttonContent}>
                <Text style={styles.buttonText}>
                  {isOtpSent ? 'Complete Verification' : 'Get Verification Code'}
                </Text>
                <Ionicons name="arrow-forward" size={18} color="#fff" style={styles.buttonArrow} />
              </View>
            )}
          </TouchableOpacity>
        </View>

        <TouchableOpacity onPress={() => navigation.navigate('Login')} style={styles.linkContainer}>
          <Text style={styles.linkTextNormal}>Already have an account? </Text>
          <Text style={styles.linkTextBold}>Login Here</Text>
        </TouchableOpacity>

      </ScrollView>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#f5f5f5' },
  scrollContainer: { flexGrow: 1, justifyContent: 'center', paddingHorizontal: 24, paddingVertical: 20 },
  headerSection: { alignItems: 'center', marginBottom: 20 },
  logoBadge: { backgroundColor: '#e8f5e9', padding: 14, borderRadius: 18, marginBottom: 12 },
  title: { fontSize: 30, fontWeight: '800', color: '#2e7d32', letterSpacing: 0.5 },
  subtitle: { fontSize: 13, color: '#666', textAlign: 'center', marginTop: 4, paddingHorizontal: 20, lineHeight: 18 },
  cardContainer: { backgroundColor: '#fff', borderRadius: 20, padding: 24, shadowColor: '#000', shadowOffset: { width: 0, height: 4 }, shadowOpacity: 0.06, shadowRadius: 12, elevation: 4 },
  cardHeaderTitle: { fontSize: 18, fontWeight: '700', color: '#333', marginBottom: 20 },
  inputWrapper: { flexDirection: 'row', alignItems: 'center', backgroundColor: '#fafafa', borderWidth: 1, borderColor: '#eef0ef', borderRadius: 12, paddingHorizontal: 14, marginBottom: 16, height: 52 },
  inputWrapperFocused: { borderColor: '#2e7d32', backgroundColor: '#fff' },
  inputDisabled: { backgroundColor: '#eee', borderColor: '#ddd', opacity: 0.6 },
  inputIcon: { marginRight: 12 },
  input: { flex: 1, fontSize: 15, color: '#222', height: '100%' },
  eyeIcon: { padding: 4 },
  otpNoticeText: { fontSize: 13, fontWeight: '600', color: '#2e7d32', marginBottom: 8, marginTop: 4 },
  otpInputText: { letterSpacing: 6, fontWeight: '700', fontSize: 16 },
  button: { backgroundColor: '#2e7d32', height: 52, borderRadius: 12, alignItems: 'center', justifyContent: 'center', marginTop: 8 },
  buttonContent: { flexDirection: 'row', alignItems: 'center' },
  buttonText: { color: '#fff', fontSize: 16, fontWeight: '700' },
  buttonArrow: { marginLeft: 8 },
  linkContainer: { flexDirection: 'row', justifyContent: 'center', alignItems: 'center', marginTop: 20, padding: 8 },
  linkTextNormal: { color: '#666', fontSize: 14 },
  linkTextBold: { color: '#2e7d32', fontSize: 14, fontWeight: '700' }
});