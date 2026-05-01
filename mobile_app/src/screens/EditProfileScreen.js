import React, { useState, useRef } from 'react';
import {
  View, Text, StyleSheet, TouchableOpacity,
  TextInput, Alert, ScrollView, ActivityIndicator, Keyboard,
  KeyboardAvoidingView, Platform,
  Animated, Easing
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import * as SecureStore from 'expo-secure-store';
import { useAuth } from '../context/AuthContext';
import { BACKEND_URL } from '../config';

const MIN_PASSWORD_LENGTH = 6;

const Field = ({ label, value, onChangeText, placeholder, secure, showToggle, onToggle }) => (
  <View style={styles.fieldWrapper}>
    <Text style={styles.fieldLabel}>{label}</Text>
    <View style={styles.inputRow}>
      <TextInput
        style={styles.input}
        value={value}
        onChangeText={onChangeText}
        placeholder={placeholder}
        placeholderTextColor="#bbb"
        secureTextEntry={secure}
        autoCapitalize="none"
      />
      {showToggle !== undefined && (
        <TouchableOpacity onPress={onToggle} style={styles.eyeBtn}>
          <Ionicons name={showToggle ? 'eye-outline' : 'eye-off-outline'} size={18} color="#999" />
        </TouchableOpacity>
      )}
    </View>
  </View>
);

export default function EditProfileScreen({ navigation }) {
  const { userInfo, updateUserInfo } = useAuth();

  const [name, setName] = useState(userInfo?.name || '');
  const [email, setEmail] = useState(userInfo?.email || '');
  const [currentPassword, setCurrentPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [showCurrent, setShowCurrent] = useState(false);
  const [showNew, setShowNew] = useState(false);
  const [showConfirm, setShowConfirm] = useState(false);
  const [profileLoading, setProfileLoading] = useState(false);
  const [passwordLoading, setPasswordLoading] = useState(false);

  const headerAnim = useRef(new Animated.Value(0)).current;
  const card1Anim = useRef(new Animated.Value(0)).current;
  const card2Anim = useRef(new Animated.Value(0)).current;

  React.useEffect(() => {
    Animated.stagger(80, [
      Animated.timing(headerAnim, { toValue: 1, duration: 350, easing: Easing.out(Easing.cubic), useNativeDriver: true }),
      Animated.timing(card1Anim, { toValue: 1, duration: 350, easing: Easing.out(Easing.cubic), useNativeDriver: true }),
      Animated.timing(card2Anim, { toValue: 1, duration: 350, easing: Easing.out(Easing.cubic), useNativeDriver: true }),
    ]).start();
  }, []);

  const animStyle = (anim, slide = 16) => ({
    opacity: anim,
    transform: [{ translateY: anim.interpolate({ inputRange: [0, 1], outputRange: [slide, 0] }) }],
  });

  const handleUpdateProfile = async () => {
    Keyboard.dismiss();

    if (!name.trim() || !email.trim()) {
      Alert.alert('Error', 'Name and email are required.');
      return;
    }

    setProfileLoading(true);
    try {
      const token = await SecureStore.getItemAsync('userToken');
      const res = await fetch(`${BACKEND_URL}/api/auth/update`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ name: name.trim(), email: email.trim() }),
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.message);

      updateUserInfo(data);
      Alert.alert('Success', 'Profile updated successfully.');
    } catch (err) {
      Alert.alert('Update Failed', err.message);
    } finally {
      setProfileLoading(false);
      Keyboard.dismiss();
    }
  };

  const handleChangePassword = async () => {
    Keyboard.dismiss();

    if (!currentPassword || !newPassword || !confirmPassword) {
      Alert.alert('Error', 'All password fields are required.');
      return;
    }
    if (newPassword !== confirmPassword) {
      Alert.alert('Error', 'New passwords do not match.');
      return;
    }
    if (newPassword.length < MIN_PASSWORD_LENGTH) {
      Alert.alert('Error', `Password must be at least ${MIN_PASSWORD_LENGTH} characters.`);
      return;
    }

    setPasswordLoading(true);
    try {
      const token = await SecureStore.getItemAsync('userToken');
      const res = await fetch(`${BACKEND_URL}/api/auth/change-password`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ currentPassword, newPassword }),
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.message);

      setCurrentPassword('');
      setNewPassword('');
      setConfirmPassword('');
      Alert.alert('Success', 'Password changed successfully.');
    } catch (err) {
      Alert.alert('Failed', err.message);
    } finally {
      setPasswordLoading(false);
      Keyboard.dismiss();
    }
  };

  return (
    <View style={styles.screen}>
      {/* Header */}
      <Animated.View style={[styles.header, animStyle(headerAnim, -10)]}>
        <TouchableOpacity style={styles.backBtn} onPress={() => navigation.goBack()}>
          <Ionicons name="chevron-back" size={20} color="#2e7d32" />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Edit Profile</Text>
        <View style={{ width: 36 }} />
      </Animated.View>

      <KeyboardAvoidingView
        style={styles.keyboardAvoidingView}
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
      >
        <ScrollView
          style={styles.scrollView}
          contentContainerStyle={styles.container}
          showsVerticalScrollIndicator={false}
          keyboardShouldPersistTaps="handled"
          keyboardDismissMode="none"
        >
          {/* Profile info card */}
          <Animated.View style={[styles.card, animStyle(card1Anim)]}>
            <Text style={styles.cardTitle}>Personal Information</Text>
            <View style={styles.cardDivider} />

            <Field
              label="Full Name"
              value={name}
              onChangeText={setName}
              placeholder="Enter your name"
            />
            <Field
              label="Email Address"
              value={email}
              onChangeText={setEmail}
              placeholder="Enter your email"
            />

            <TouchableOpacity
              style={[styles.saveBtn, profileLoading && styles.saveBtnDisabled]}
              onPress={handleUpdateProfile}
              disabled={profileLoading}
              activeOpacity={0.8}
            >
              {profileLoading
                ? <ActivityIndicator color="#fff" />
                : <Text style={styles.saveBtnText}>Save Changes</Text>
              }
            </TouchableOpacity>
          </Animated.View>

          {/* Change password card */}
          <Animated.View style={[styles.card, animStyle(card2Anim)]}>
            <Text style={styles.cardTitle}>Change Password</Text>
            <View style={styles.cardDivider} />

            <Field
              label="Current Password"
              value={currentPassword}
              onChangeText={setCurrentPassword}
              placeholder="Enter current password"
              secure={!showCurrent}
              showToggle={showCurrent}
              onToggle={() => setShowCurrent(p => !p)}
            />
            <Field
              label="New Password"
              value={newPassword}
              onChangeText={setNewPassword}
              placeholder={`Min. ${MIN_PASSWORD_LENGTH} characters`}
              secure={!showNew}
              showToggle={showNew}
              onToggle={() => setShowNew(p => !p)}
            />
            <Field
              label="Confirm New Password"
              value={confirmPassword}
              onChangeText={setConfirmPassword}
              placeholder="Repeat new password"
              secure={!showConfirm}
              showToggle={showConfirm}
              onToggle={() => setShowConfirm(p => !p)}
            />

            <TouchableOpacity
              style={[
                styles.saveBtn,
                (passwordLoading || newPassword.length < MIN_PASSWORD_LENGTH || newPassword !== confirmPassword) && styles.saveBtnDisabled,
              ]}
              onPress={handleChangePassword}
              disabled={passwordLoading || newPassword.length < MIN_PASSWORD_LENGTH || newPassword !== confirmPassword}
              activeOpacity={0.8}
            >
              {passwordLoading
                ? <ActivityIndicator color="#fff" />
                : <Text style={styles.saveBtnText}>Update Password</Text>
              }
            </TouchableOpacity>
          </Animated.View>
        </ScrollView>
      </KeyboardAvoidingView>
    </View>
  );
}

const styles = StyleSheet.create({
  screen: {
    flex: 1,
    backgroundColor: '#f7faf7',
  },
  scrollView: {
    flex: 1,
  },
  keyboardAvoidingView: {
    flex: 1,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingTop: 35,
    paddingBottom: 10,
    backgroundColor: '#f7faf7',
  },
  backBtn: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: '#e8f5e9',
    justifyContent: 'center',
    alignItems: 'center',
  },
  headerTitle: {
    flex: 1,
    textAlign: 'center',
    fontSize: 17,
    fontWeight: '600',
    color: '#1b5e20',
  },
  container: {
    padding: 16,
    paddingBottom: 110,
  },
  card: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 18,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: '#e8f0e8',
  },
  cardTitle: {
    fontSize: 15,
    fontWeight: '700',
    color: '#1b5e20',
  },
  cardDivider: {
    height: 1,
    backgroundColor: '#f0f4f0',
    marginVertical: 14,
  },
  fieldWrapper: {
    marginBottom: 14,
  },
  fieldLabel: {
    fontSize: 11,
    color: '#999',
    textTransform: 'uppercase',
    letterSpacing: 0.8,
    marginBottom: 6,
  },
  inputRow: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#f7faf7',
    borderRadius: 10,
    borderWidth: 1,
    borderColor: '#e0ece0',
  },
  input: {
    flex: 1,
    paddingHorizontal: 14,
    paddingVertical: 12,
    fontSize: 15,
    color: '#1a1a1a',
  },
  eyeBtn: {
    paddingHorizontal: 12,
  },
  saveBtn: {
    backgroundColor: '#2e7d32',
    paddingVertical: 13,
    borderRadius: 10,
    alignItems: 'center',
    marginTop: 4,
  },
  saveBtnDisabled: {
    backgroundColor: '#a5d6a7',
  },
  saveBtnText: {
    color: '#fff',
    fontWeight: '600',
    fontSize: 15,
  },
});