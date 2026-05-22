import React, { useState, useRef } from "react";
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  ActivityIndicator,
  KeyboardAvoidingView,
  Platform,
  ScrollView,
  Modal,
  ToastAndroid,
  Alert,
  Image,
} from "react-native";
import { useAuth } from "../context/AuthContext";
import {
  loginUser,
  sendOtpToEmail,
  executePasswordReset,
} from "../services/authService";
import { Ionicons } from "@expo/vector-icons";

export default function LoginScreen({ navigation }) {
  const { login } = useAuth();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [showModalPassword, setShowModalPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [focusedField, setFocusedField] = useState(null);

  // Recovery States
  const [forgotModalVisible, setForgotModalVisible] = useState(false);
  const [recoveryEmail, setRecoveryEmail] = useState("");
  const [recoveryOtp, setRecoveryOtp] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [isRecoveryOtpSent, setIsRecoveryOtpSent] = useState(false);
  const [modalLoading, setModalLoading] = useState(false);

  const passwordFieldRef = useRef(null);

  // Fallback notification wrapper for cross-platform alerts
  const showNotification = (msg) => {
    if (Platform.OS === "android") {
      ToastAndroid.showWithGravityAndOffset(
        msg,
        ToastAndroid.LONG,
        ToastAndroid.BOTTOM,
        0,
        50,
      );
    } else {
      Alert.alert("AgriGuard", msg);
    }
  };

  const handleLogin = async () => {
    if (!email.trim() || !password.trim()) {
      showNotification("Please enter your email and password keys.");
      return;
    }
    setLoading(true);
    try {
      const data = await loginUser(email.trim(), password);
      await login(data.token, data.user);
    } catch (err) {
      const message =
        err.response?.data?.message || "Login rejected. Verify credentials.";
      showNotification(message);
    } finally {
      setLoading(false);
    }
  };

  const handleRequestRecoveryOTP = async () => {
    if (!recoveryEmail.trim()) {
      showNotification("Please enter your registered email address.");
      return;
    }
    setModalLoading(true);
    try {
      await sendOtpToEmail(recoveryEmail.trim(), "reset");
      showNotification("Reset key delivered! Check your inbox.");
      setIsRecoveryOtpSent(true);
    } catch (err) {
      const msg = err.response?.data?.message || "Account profile untraceable.";
      showNotification(msg);
    } finally {
      setModalLoading(false);
    }
  };

  const handleExecuteReset = async () => {
    if (!recoveryOtp.trim() || !newPassword.trim()) {
      showNotification("Please complete all reset query boxes.");
      return;
    }
    if (newPassword.length < 6) {
      showNotification("New password must be at least 6 characters long.");
      return;
    }
    setModalLoading(true);
    try {
      const res = await executePasswordReset(
        recoveryEmail.trim(),
        recoveryOtp.trim(),
        newPassword.trim(),
      );
      showNotification("Password updated successfully! Proceed to login.");
      closeRecoveryModal();
    } catch (err) {
      const msg =
        err.response?.data?.message || "Reset rejected. Token expired.";
      showNotification(msg);
    } finally {
      setModalLoading(false);
    }
  };

  const closeRecoveryModal = () => {
    setForgotModalVisible(false);
    setRecoveryEmail("");
    setRecoveryOtp("");
    setNewPassword("");
    setIsRecoveryOtpSent(false);
    setShowModalPassword(false);
  };

  return (
    <KeyboardAvoidingView
      style={styles.container}
      behavior={Platform.OS === "ios" ? "padding" : "height"}
      keyboardVerticalOffset={Platform.OS === "ios" ? 64 : 0}
    >
      <ScrollView
        contentContainerStyle={styles.scrollContainer}
        showsVerticalScrollIndicator={false}
        keyboardShouldPersistTaps="handled"
      >
        {/* Render premium logo asset from assets/logo-light.png */}
        <View style={styles.headerSection}>
          <Image
            source={require("../../assets/logo-light.png")}
            style={styles.brandingLogo}
            resizeMode="contain"
          />
          <Text style={styles.subtitle}>
            Smart AI Disease Diagnostics & Management
          </Text>
        </View>

        <View style={styles.cardContainer}>
          <Text style={styles.cardHeaderTitle}>Account Login</Text>

          {/* Email input field */}
          <View
            style={[
              styles.inputWrapper,
              focusedField === "email" && styles.inputWrapperFocused,
            ]}
          >
            <Ionicons
              name="mail-outline"
              size={20}
              color={focusedField === "email" ? "#2e7d32" : "#888"}
              style={styles.inputIcon}
            />
            <TextInput
              style={styles.input}
              placeholder="Email Address"
              placeholderTextColor="#a0a0a0"
              keyboardType="email-address"
              autoCapitalize="none"
              value={email}
              onChangeText={setEmail}
              onFocus={() => setFocusedField("email")}
              onBlur={() => setFocusedField(null)}
              returnKeyType="next"
              onSubmitEditing={() => passwordFieldRef.current?.focus()}
              blurOnSubmit={false}
            />
          </View>

          {/* Password field with dynamic visibility toggle */}
          <View
            style={[
              styles.inputWrapper,
              focusedField === "password" && styles.inputWrapperFocused,
            ]}
          >
            <Ionicons
              name="lock-closed-outline"
              size={20}
              color={focusedField === "password" ? "#2e7d32" : "#888"}
              style={styles.inputIcon}
            />
            <TextInput
              ref={passwordFieldRef}
              style={styles.input}
              placeholder="Password"
              placeholderTextColor="#a0a0a0"
              secureTextEntry={!showPassword}
              value={password}
              onChangeText={setPassword}
              onFocus={() => setFocusedField("password")}
              onBlur={() => setFocusedField(null)}
              returnKeyType="done"
              onSubmitEditing={handleLogin}
            />
            <TouchableOpacity
              onPress={() => setShowPassword(!showPassword)}
              style={styles.eyeIcon}
            >
              <Ionicons
                name={showPassword ? "eye-off-outline" : "eye-outline"}
                size={20}
                color="#888"
              />
            </TouchableOpacity>
          </View>

          <TouchableOpacity
            onPress={() => setForgotModalVisible(true)}
            style={styles.forgotLinkContainer}
          >
            <Text style={styles.forgotLinkText}>Forgot Password?</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={styles.button}
            onPress={handleLogin}
            disabled={loading}
          >
            {loading ? (
              <ActivityIndicator color="#fff" />
            ) : (
              <View style={styles.buttonContent}>
                <Text style={styles.buttonText}>Login</Text>
                <Ionicons
                  name="log-in-outline"
                  size={18}
                  color="#fff"
                  style={styles.buttonArrow}
                />
              </View>
            )}
          </TouchableOpacity>
        </View>

        <TouchableOpacity
          onPress={() => navigation.navigate("Register")}
          style={styles.linkContainer}
        >
          <Text style={styles.linkTextNormal}>New to AgriGuard? </Text>
          <Text style={styles.linkTextBold}>Register Here</Text>
        </TouchableOpacity>
      </ScrollView>

      {/* Password reset workflows overlay */}
      <Modal
        visible={forgotModalVisible}
        animationType="slide"
        transparent={true}
        onRequestClose={closeRecoveryModal}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalContentCard}>
            <View style={styles.modalHeaderRow}>
              <TouchableOpacity
                onPress={closeRecoveryModal}
                style={styles.modalLeftClose}
              >
                <Ionicons name="close-circle" size={26} color="#888" />
              </TouchableOpacity>
              <Text style={styles.modalTitle}>Reset Password</Text>
              <View style={{ width: 26 }} />
            </View>

            {!isRecoveryOtpSent ? (
              <View>
                <Text style={styles.modalInstruction}>
                  Enter your account email below. We will send a secure 6-digit
                  code to authorize a password change.
                </Text>
                <View style={styles.inputWrapper}>
                  <Ionicons
                    name="mail-outline"
                    size={20}
                    color="#888"
                    style={styles.inputIcon}
                  />
                  <TextInput
                    style={styles.input}
                    placeholder="Account Email Address"
                    placeholderTextColor="#a0a0a0"
                    keyboardType="email-address"
                    autoCapitalize="none"
                    value={recoveryEmail}
                    onChangeText={setRecoveryEmail}
                  />
                </View>
                <TouchableOpacity
                  style={styles.button}
                  onPress={handleRequestRecoveryOTP}
                  disabled={modalLoading}
                >
                  {modalLoading ? (
                    <ActivityIndicator color="#fff" />
                  ) : (
                    <Text style={styles.buttonText}>Get Reset Code</Text>
                  )}
                </TouchableOpacity>
              </View>
            ) : (
              <View>
                <Text style={styles.modalInstruction}>
                  Type the code sent to your email inbox and establish your new
                  login password parameters.
                </Text>

                <View style={styles.inputWrapper}>
                  <Ionicons
                    name="shield-checkmark-outline"
                    size={20}
                    color="#888"
                    style={styles.inputIcon}
                  />
                  <TextInput
                    style={styles.input}
                    placeholder="6-Digit Verification Code"
                    placeholderTextColor="#a0a0a0"
                    keyboardType="number-pad"
                    maxLength={6}
                    value={recoveryOtp}
                    onChangeText={setRecoveryOtp}
                  />
                </View>

                <View style={styles.inputWrapper}>
                  <Ionicons
                    name="lock-open-outline"
                    size={20}
                    color="#888"
                    style={styles.inputIcon}
                  />
                  <TextInput
                    style={styles.input}
                    placeholder="Type New Password"
                    placeholderTextColor="#a0a0a0"
                    secureTextEntry={!showModalPassword}
                    value={newPassword}
                    onChangeText={setNewPassword}
                  />
                  <TouchableOpacity
                    onPress={() => setShowModalPassword(!showModalPassword)}
                    style={styles.eyeIcon}
                  >
                    <Ionicons
                      name={
                        showModalPassword ? "eye-off-outline" : "eye-outline"
                      }
                      size={20}
                      color="#888"
                    />
                  </TouchableOpacity>
                </View>

                <TouchableOpacity
                  style={styles.button}
                  onPress={handleExecuteReset}
                  disabled={modalLoading}
                >
                  {modalLoading ? (
                    <ActivityIndicator color="#fff" />
                  ) : (
                    <Text style={styles.buttonText}>Save New Password</Text>
                  )}
                </TouchableOpacity>
              </View>
            )}
          </View>
        </View>
      </Modal>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#f5f5f5" },
  scrollContainer: {
    flexGrow: 1,
    justifyContent: "center",
    paddingHorizontal: 24,
    paddingVertical: 20,
  },
  headerSection: { 
    alignItems: 'center', 
    marginBottom: 32,
    marginTop: 40 
  },
  brandingLogo: { 
    width: 240, 
    height: 65, 
    marginBottom: 8,
    // Mix-blend fallback: forces the image to drop asset-level container artifacts
    backgroundColor: 'transparent'
  },
  subtitle: { 
    fontSize: 14, 
    color: '#4e5451', 
    textAlign: 'center', 
    fontWeight: '600', 
    paddingHorizontal: 10, 
    lineHeight: 20,
    letterSpacing: 0.2
  },
  cardContainer: {
    backgroundColor: "#fff",
    borderRadius: 20,
    padding: 24,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.06,
    shadowRadius: 12,
    elevation: 4,
  },
  cardHeaderTitle: {
    fontSize: 18,
    fontWeight: "700",
    color: "#333",
    marginBottom: 20,
  },
  inputWrapper: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "#fafafa",
    borderWidth: 1,
    borderColor: "#eef0ef",
    borderRadius: 12,
    paddingHorizontal: 14,
    marginBottom: 16,
    height: 52,
  },
  inputWrapperFocused: { borderColor: "#2e7d32", backgroundColor: "#fff" },
  inputIcon: { marginRight: 12 },
  input: { flex: 1, fontSize: 15, color: "#222", height: "100%" },
  eyeIcon: { padding: 4 },
  forgotLinkContainer: {
    alignSelf: "flex-end",
    marginBottom: 20,
    marginTop: -4,
    padding: 2,
  },
  forgotLinkText: { color: "#2e7d32", fontSize: 13, fontWeight: "600" },
  button: {
    backgroundColor: "#2e7d32",
    height: 52,
    borderRadius: 12,
    alignItems: "center",
    justifyContent: "center",
  },
  buttonContent: { flexDirection: "row", alignItems: "center" },
  buttonText: { color: "#fff", fontSize: 16, fontWeight: "700" },
  buttonArrow: { marginLeft: 8 },
  linkContainer: {
    flexDirection: "row",
    justifyContent: "center",
    alignItems: "center",
    marginTop: 24,
    padding: 8,
  },
  linkTextNormal: { color: "#666", fontSize: 14 },
  linkTextBold: { color: "#2e7d32", fontSize: 14, fontWeight: "700" },
  modalOverlay: {
    flex: 1,
    backgroundColor: "rgba(0,0,0,0.4)",
    justifyContent: "center",
    paddingHorizontal: 20,
  },
  modalContentCard: {
    backgroundColor: "#fff",
    borderRadius: 20,
    padding: 24,
    shadowColor: "#000",
    shadowOpacity: 0.1,
    shadowRadius: 10,
    elevation: 5,
  },
  modalHeaderRow: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: 18,
  },
  modalLeftClose: { marginRight: 12, padding: 2 },
  modalTitle: {
    flex: 1,
    fontSize: 20,
    fontWeight: "700",
    color: "#2e7d32",
    textAlign: "left",
  },
  modalInstruction: {
    fontSize: 13,
    color: "#666",
    marginBottom: 16,
    lineHeight: 18,
  },
});
