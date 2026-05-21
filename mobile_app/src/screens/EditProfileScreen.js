import React, { useState, useRef } from "react";
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  TextInput,
  ActivityIndicator,
  Keyboard,
  KeyboardAvoidingView,
  Platform,
  ScrollView,
  Animated,
  Easing,
  ToastAndroid,
  Alert,
  StatusBar,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import * as SecureStore from "expo-secure-store";
import { useAuth } from "../context/AuthContext";
import { BACKEND_URL } from "../config";

const MIN_PASSWORD_LENGTH = 6;

// Upgraded Field component supporting forwarding focus references
const Field = React.forwardRef(
  (
    {
      label,
      value,
      onChangeText,
      placeholder,
      secure,
      showToggle,
      onToggle,
      editable = true,
      ...props
    },
    ref,
  ) => (
    <View style={styles.fieldWrapper}>
      <Text style={styles.fieldLabel}>{label}</Text>
      <View style={[styles.inputRow, !editable && styles.inputRowDisabled]}>
        <TextInput
          ref={ref}
          style={[styles.input, !editable && styles.inputDisabled]}
          value={value}
          onChangeText={onChangeText}
          placeholder={placeholder}
          placeholderTextColor="#bbb"
          secureTextEntry={secure}
          autoCapitalize="none"
          editable={editable}
          blurOnSubmit={false}
          {...props}
        />
        {showToggle !== undefined && (
          <TouchableOpacity onPress={onToggle} style={styles.eyeBtn}>
            <Ionicons
              name={showToggle ? "eye-outline" : "eye-off-outline"}
              size={18}
              color="#999"
            />
          </TouchableOpacity>
        )}
      </View>
    </View>
  ),
);

export default function EditProfileScreen({ navigation }) {
  const { userInfo, updateUserInfo } = useAuth();

  const [name, setName] = useState(userInfo?.name || "");
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");

  const [showCurrent, setShowCurrent] = useState(false);
  const [showNew, setShowNew] = useState(false);
  const [showConfirm, setShowConfirm] = useState(false);

  const [profileLoading, setProfileLoading] = useState(false);
  const [passwordLoading, setPasswordLoading] = useState(false);

  // Focus Input Pointer Refs
  const nameInputRef = useRef(null);
  const currentPasswordRef = useRef(null);
  const newPasswordRef = useRef(null);
  const confirmPasswordRef = useRef(null);

  const headerAnim = useRef(new Animated.Value(0)).current;
  const card1Anim = useRef(new Animated.Value(0)).current;
  const card2Anim = useRef(new Animated.Value(0)).current;

  React.useEffect(() => {
    Animated.stagger(80, [
      Animated.timing(headerAnim, {
        toValue: 1,
        duration: 350,
        easing: Easing.out(Easing.cubic),
        useNativeDriver: true,
      }),
      Animated.timing(card1Anim, {
        toValue: 1,
        duration: 350,
        easing: Easing.out(Easing.cubic),
        useNativeDriver: true,
      }),
      Animated.timing(card2Anim, {
        toValue: 1,
        duration: 350,
        easing: Easing.out(Easing.cubic),
        useNativeDriver: true,
      }),
    ]).start();
  }, []);

  const showNotification = (msg) => {
    if (Platform.OS === "android") {
      ToastAndroid.showWithGravityAndOffset(
        msg,
        ToastAndroid.SHORT,
        ToastAndroid.BOTTOM,
        0,
        50,
      );
    } else {
      Alert.alert("AgriGuard", msg);
    }
  };

  const animStyle = (anim, slide = 16) => ({
    opacity: anim,
    transform: [
      {
        translateY: anim.interpolate({
          inputRange: [0, 1],
          outputRange: [slide, 0],
        }),
      },
    ],
  });

  const handleUpdateProfile = async () => {
    Keyboard.dismiss();
    if (!name.trim()) {
      showNotification("Name is a required field.");
      return;
    }

    setProfileLoading(true);
    try {
      const token = await SecureStore.getItemAsync("userToken");
      const res = await fetch(`${BACKEND_URL}/api/auth/update`, {
        method: "PUT",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ name: name.trim(), email: userInfo.email }),
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.message);

      updateUserInfo(data);
      showNotification("Name updated successfully!");
    } catch (err) {
      showNotification(err.message);
    } finally {
      setProfileLoading(false);
    }
  };

  const handleChangePassword = async () => {
    Keyboard.dismiss();
    if (!currentPassword || !newPassword || !confirmPassword) {
      showNotification("All password fields are required.");
      return;
    }
    if (newPassword !== confirmPassword) {
      showNotification("New passwords do not match.");
      return;
    }
    if (newPassword.length < MIN_PASSWORD_LENGTH) {
      showNotification(
        `Password must be at least ${MIN_PASSWORD_LENGTH} characters.`,
      );
      return;
    }

    setPasswordLoading(true);
    try {
      const token = await SecureStore.getItemAsync("userToken");
      const res = await fetch(`${BACKEND_URL}/api/auth/change-password`, {
        method: "PUT",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ currentPassword, newPassword }),
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.message);

      setCurrentPassword("");
      setNewPassword("");
      setConfirmPassword("");
      showNotification("Password updated successfully!");
    } catch (err) {
      showNotification(err.message);
    } finally {
      setPasswordLoading(false);
    }
  };

  return (
    <View style={styles.screen}>
      {/* Header Layout */}
      <Animated.View style={[styles.header, animStyle(headerAnim, -10)]}>
        <TouchableOpacity
          style={styles.backBtn}
          onPress={() => navigation.goBack()}
        >
          <Ionicons name="chevron-back" size={20} color="#2e7d32" />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Modify Account Settings</Text>
        <View style={{ width: 36 }} />
      </Animated.View>

      <KeyboardAvoidingView
        style={styles.keyboardAvoidingView}
        behavior={Platform.OS === "ios" ? "padding" : "height"}
      >
        <ScrollView
          style={styles.scrollView}
          contentContainerStyle={styles.container}
          showsVerticalScrollIndicator={false}
          keyboardShouldPersistTaps="handled"
        >
          {/* Personal Information Layout Card */}
          <Animated.View style={[styles.card, animStyle(card1Anim)]}>
            <Text style={styles.cardTitle}>Personal Profile Data</Text>
            <View style={styles.cardDivider} />

            <Field
              ref={nameInputRef}
              label="Full Name"
              value={name}
              onChangeText={setName}
              placeholder="Enter your full name"
              returnKeyType="done"
              onSubmitEditing={() => Keyboard.dismiss()}
            />

            <Field
              label="Email Address (Locked)"
              value={userInfo?.email || "farmer@agriguard.com"}
              editable={false}
            />

            <TouchableOpacity
              style={styles.saveBtn}
              onPress={handleUpdateProfile}
              disabled={profileLoading}
              activeOpacity={0.8}
            >
              {profileLoading ? (
                <ActivityIndicator color="#fff" />
              ) : (
                <Text style={styles.saveBtnText}>Save Personal Changes</Text>
              )}
            </TouchableOpacity>
          </Animated.View>

          {/* Change Password Layout Card */}
          <Animated.View style={[styles.card, animStyle(card2Anim)]}>
            <Text style={styles.cardTitle}>Update Password</Text>
            <View style={styles.cardDivider} />

            <Field
              ref={currentPasswordRef}
              label="Current Password"
              value={currentPassword}
              onChangeText={setCurrentPassword}
              placeholder="Type current password"
              secure={!showCurrent}
              showToggle={showCurrent}
              onToggle={() => setShowCurrent((p) => !p)}
              returnKeyType="next"
              onSubmitEditing={() => newPasswordRef.current?.focus()}
            />

            <Field
              ref={newPasswordRef}
              label="New Password"
              value={newPassword}
              onChangeText={setNewPassword}
              placeholder={`Min. ${MIN_PASSWORD_LENGTH} characters`}
              secure={!showNew}
              showToggle={showNew}
              onToggle={() => setShowNew((p) => !p)}
              returnKeyType="next"
              onSubmitEditing={() => confirmPasswordRef.current?.focus()}
            />

            <Field
              ref={confirmPasswordRef}
              label="Confirm New Password"
              value={confirmPassword}
              onChangeText={setConfirmPassword}
              placeholder="Repeat new password"
              secure={!showConfirm}
              showToggle={showConfirm}
              onToggle={() => setShowConfirm((p) => !p)}
              returnKeyType="done"
              onSubmitEditing={handleChangePassword}
            />

            <TouchableOpacity
              style={styles.saveBtn}
              onPress={handleChangePassword}
              disabled={passwordLoading}
              activeOpacity={0.8}
            >
              {passwordLoading ? (
                <ActivityIndicator color="#fff" />
              ) : (
                <Text style={styles.saveBtnText}>Update Password</Text>
              )}
            </TouchableOpacity>
          </Animated.View>
        </ScrollView>
      </KeyboardAvoidingView>
    </View>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: "#f5f5f5" },
  scrollView: { flex: 1 },
  keyboardAvoidingView: { flex: 1 },
  // FIXED TOP PADDING: Integrated safe bar padding variables cleanly across platforms
  header: {
    flexDirection: "row",
    alignItems: "center",
    paddingHorizontal: 16,
    paddingTop:
      Platform.OS === "ios"
        ? 52
        : StatusBar.currentHeight
          ? StatusBar.currentHeight + 12
          : 12,
    paddingBottom: 14,
    backgroundColor: "#fff",
    borderBottomWidth: 1,
    borderBottomColor: "#eef0ef",
  },
  backBtn: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: "#e8f5e9",
    justifyContent: "center",
    alignItems: "center",
  },
  headerTitle: {
    flex: 1,
    textAlign: "center",
    fontSize: 16,
    fontWeight: "800",
    color: "#2e7d32",
  },
  container: { padding: 16, paddingBottom: 40 },
  card: {
    backgroundColor: "#fff",
    borderRadius: 20,
    padding: 20,
    marginBottom: 16,
    elevation: 2,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.04,
    shadowRadius: 8,
  },
  cardTitle: { fontSize: 15, fontWeight: "800", color: "#333" },
  cardDivider: { height: 1, backgroundColor: "#f5f5f5", marginVertical: 14 },
  fieldWrapper: { marginBottom: 16 },
  fieldLabel: {
    fontSize: 11,
    color: "#999",
    fontWeight: "800",
    textTransform: "uppercase",
    letterSpacing: 0.8,
    marginBottom: 6,
  },
  inputRow: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "#fafafa",
    borderRadius: 12,
    borderWidth: 1,
    borderColor: "#eef0ef",
    height: 50,
  },
  inputRowDisabled: { backgroundColor: "#f0f0f0", borderColor: "#e0e0e0" },
  input: {
    flex: 1,
    paddingHorizontal: 14,
    fontSize: 15,
    color: "#222",
    height: "100%",
  },
  inputDisabled: { color: "#888" },
  eyeBtn: { paddingHorizontal: 12 },
  saveBtn: {
    backgroundColor: "#2e7d32",
    height: 48,
    borderRadius: 12,
    alignItems: "center",
    justifyContent: "center",
    marginTop: 4,
  },
  saveBtnText: { color: "#fff", fontWeight: "700", fontSize: 15 },
});
