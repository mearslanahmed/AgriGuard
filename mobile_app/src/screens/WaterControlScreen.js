  import React, { useState, useEffect, useRef } from "react";
  import {
    View,
    Text,
    StyleSheet,
    TouchableOpacity,
    ScrollView,
    ActivityIndicator,
    Animated,
    Easing,
    Platform,
    StatusBar,
    Switch,
    RefreshControl,
    Modal,
  } from "react-native";
  import * as SecureStore from "expo-secure-store";
  import { Ionicons } from "@expo/vector-icons";
  import { BACKEND_URL } from "../config";

  export default function WaterControlScreen({ navigation }) {
    const [pumpOn, setPumpOn] = useState(false);
    const [moisture, setMoisture] = useState(null);
    const [autoMode, setAutoMode] = useState(false);
    const [loading, setLoading] = useState(false);
    const [autoLoading, setAutoLoading] = useState(false);
    const [statusLoading, setStatusLoading] = useState(true);
    const [refreshing, setRefreshing] = useState(false);
    const [lastUpdated, setLastUpdated] = useState(null);
    const [moistureUpdated, setMoistureUpdated] = useState(null);
    const [connectionStatus, setConnectionStatus] = useState("checking");

    // Modal state — single modal handles all cases via config
    const [modal, setModal] = useState({
      visible: false,
      title: "",
      message: "",
      type: "error", // 'error' | 'confirm'
      onConfirm: null,
    });

    const headerAnim = useRef(new Animated.Value(0)).current;
    const cardAnim = useRef(new Animated.Value(0)).current;
    const infoAnim = useRef(new Animated.Value(0)).current;

    useEffect(() => {
      Animated.stagger(80, [
        Animated.timing(headerAnim, {
          toValue: 1,
          duration: 350,
          easing: Easing.out(Easing.cubic),
          useNativeDriver: true,
        }),
        Animated.timing(cardAnim, {
          toValue: 1,
          duration: 350,
          easing: Easing.out(Easing.cubic),
          useNativeDriver: true,
        }),
        Animated.timing(infoAnim, {
          toValue: 1,
          duration: 350,
          easing: Easing.out(Easing.cubic),
          useNativeDriver: true,
        }),
      ]).start();

      fetchStatus();
    }, []);

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

    const showError = (title, message) => {
      setModal({ visible: true, title, message, type: "error", onConfirm: null });
    };

    const showConfirm = (title, message, onConfirm) => {
      setModal({ visible: true, title, message, type: "confirm", onConfirm });
    };

    const closeModal = () => setModal((m) => ({ ...m, visible: false }));

    const fetchStatus = async () => {
    setStatusLoading(true);
    try {
      const token = await SecureStore.getItemAsync("userToken");
      if (!token) {
            console.log("No token found!");
            setConnectionStatus("offline");
            return;
        }
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000);

      const response = await fetch(`${BACKEND_URL}/api/esp/wroom/status`, {
        headers: { Authorization: `Bearer ${token}` },
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (response.ok) {
        const data = await response.json();
        // console.log("Full JSON received:", data);
        setPumpOn(data.pump === "on");
        setMoisture(data.moisture);
        setAutoMode(data.autoMode);
        setLastUpdated(new Date());
        setMoistureUpdated(new Date());
        setConnectionStatus("online");
      } else {
        setConnectionStatus("offline");
      }
    } catch (err) {
      setConnectionStatus("offline");
      console.log("WROOM status fetch failed:", err.message);
    } finally {
      setStatusLoading(false);
      setRefreshing(false);
    }
  };

    const onRefresh = () => {
      setRefreshing(true);
      fetchStatus();
    };

    const togglePump = () => {
      if (connectionStatus === "offline") {
        showError(
          "Device Offline",
          "Cannot reach the irrigation controller. Make sure the device is powered on and connected to the network.",
        );
        return;
      }

      const action = pumpOn ? "turn OFF" : "turn ON";
      showConfirm(
        "Confirm Action",
        `Are you sure you want to ${action} the water pump?`,
        async () => {
          closeModal();
          setLoading(true);
          try {
            const token = await SecureStore.getItemAsync("userToken");
            const endpoint = pumpOn
              ? "/api/esp/wroom/pump/off"
              : "/api/esp/wroom/pump/on";
            const response = await fetch(`${BACKEND_URL}${endpoint}`, {
              headers: { Authorization: `Bearer ${token}` },
            });
            if (response.ok) {
              setPumpOn(!pumpOn);
              setLastUpdated(new Date());
            } else {
              showError(
                "Command Failed",
                "Failed to send command to the device. Please try again.",
              );
            }
          } catch (err) {
            showError(
              "Connection Error",
              "Could not reach the irrigation controller.",
            );
          } finally {
            setLoading(false);
          }
        },
      );
    };

    const toggleAutoMode = async (value) => {
    if (connectionStatus === "offline") {
      showError("Device Offline", "Cannot reach the irrigation controller.");
      return;
    }
    setAutoLoading(true);
    try {
      const token = await SecureStore.getItemAsync("userToken");
      const endpoint = value ? "/api/esp/wroom/auto/on" : "/api/esp/wroom/auto/off";

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000);

      const response = await fetch(`${BACKEND_URL}${endpoint}`, {
        headers: { Authorization: `Bearer ${token}` },
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (response.ok) {
        setAutoMode(value);
        if (!value) setPumpOn(false);
      } else {
        showError("Failed", "Could not update auto mode. Try again.");
      }
    } catch (err) {
      showError("Connection Error", "Could not reach the irrigation controller.");
    } finally {
      setAutoLoading(false);
    }
  };

    const getMoistureColor = (val) => {
      if (val === null) return "#bbb";
      if (val < 30) return "#e53935";
      if (val < 60) return "#f57c00";
      return "#2e7d32";
    };

    const getMoistureLabel = (val) => {
      if (val === null) return "Unknown";
      if (val < 30) return "Dry";
      if (val < 60) return "Moderate";
      return "Moist";
    };

    const getMoistureAdvice = (val) => {
      if (val === null) return null;
      if (val < 30) return "Soil is dry — consider watering your crop.";
      if (val < 60) return "Soil moisture is moderate — monitor regularly.";
      return "Soil is well moistened — no watering needed.";
    };

    const formatTime = (date) => {
      if (!date) return "Never";
      return date.toLocaleTimeString("en-PK", {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
      });
    };

    return (
      <View style={styles.screen}>
        <StatusBar barStyle="dark-content" backgroundColor="#fff" />

        <Animated.View style={[styles.header, animStyle(headerAnim, -10)]}>
          <TouchableOpacity
            style={styles.backBtn}
            onPress={() => navigation.goBack()}
          >
            <Ionicons name="chevron-back" size={20} color="#2e7d32" />
          </TouchableOpacity>
          <Text style={styles.headerTitle}>Water Management</Text>
          <View style={{ width: 36 }} />
        </Animated.View>

        <ScrollView
          style={styles.scrollView}
          contentContainerStyle={styles.content}
          refreshControl={
            <RefreshControl
              refreshing={refreshing}
              onRefresh={onRefresh}
              colors={["#2e7d32"]}
              tintColor="#2e7d32"
            />
          }
        >
          <Animated.View style={animStyle(cardAnim)}>
            <View
              style={[
                styles.statusBar,
                {
                  backgroundColor:
                    connectionStatus === "online" ? "#e8f5e9" : "#fdecea",
                },
              ]}
            >
              <View
                style={[
                  styles.statusDot,
                  {
                    backgroundColor:
                      connectionStatus === "online" ? "#2e7d32" : "#e53935",
                  },
                ]}
              />
              <Text
                style={[
                  styles.statusText,
                  {
                    color: connectionStatus === "online" ? "#2e7d32" : "#e53935",
                  },
                ]}
              >
                {connectionStatus === "checking" && "Connecting to device..."}
                {connectionStatus === "online" && "Irrigation Controller Online"}
                {connectionStatus === "offline" &&
                  "Controller Offline — Check Device"}
              </Text>
            </View>
          </Animated.View>

          {statusLoading ? (
            <ActivityIndicator
              size="large"
              color="#2e7d32"
              style={{ marginTop: 60 }}
            />
          ) : (
            <>
              <Animated.View style={[styles.card, animStyle(cardAnim)]}>
                <Text style={styles.cardTitle}>Irrigation Pump</Text>
                <View style={styles.cardDivider} />

                <View
                  style={[
                    styles.pumpIndicator,
                    { backgroundColor: pumpOn ? "#e8f5e9" : "#f5f5f5" },
                  ]}
                >
                  <View
                    style={[
                      styles.pumpDot,
                      { backgroundColor: pumpOn ? "#2e7d32" : "#bbb" },
                    ]}
                  />
                  <Text
                    style={[
                      styles.pumpStatus,
                      { color: pumpOn ? "#2e7d32" : "#888" },
                    ]}
                  >
                    {pumpOn ? "RUNNING" : "STOPPED"}
                  </Text>
                </View>

                <TouchableOpacity
                  style={[
                    styles.toggleButton,
                    { backgroundColor: pumpOn ? "#e53935" : "#2e7d32" },
                    (loading || autoMode) && styles.buttonDisabled,
                  ]}
                  onPress={togglePump}
                  disabled={loading || autoMode}
                >
                  {loading ? (
                    <ActivityIndicator color="#fff" />
                  ) : (
                    <Text style={styles.toggleText}>
                      {autoMode
                        ? "Auto Mode Active"
                        : pumpOn
                          ? "Turn Off Pump"
                          : "Turn On Pump"}
                    </Text>
                  )}
                </TouchableOpacity>

                <Text style={styles.lastUpdated}>
                  Last updated: {formatTime(lastUpdated)}
                </Text>
              </Animated.View>

              <Animated.View style={[styles.card, animStyle(cardAnim)]}>
                <View style={styles.autoRow}>
                  <View style={styles.autoTextGroup}>
                    <Text style={styles.cardTitle}>Auto Watering</Text>
                    <Text style={styles.autoSubtitle}>
                      Automatically monitors soil and controls pump based on
                      moisture level
                    </Text>
                  </View>
                  {autoLoading ? (
                    <ActivityIndicator color="#2e7d32" />
                  ) : (
                    <Switch
                      value={autoMode}
                      onValueChange={toggleAutoMode}
                      trackColor={{ false: "#e0e0e0", true: "#a5d6a7" }}
                      thumbColor={autoMode ? "#2e7d32" : "#f5f5f5"}
                    />
                  )}
                </View>
              </Animated.View>

              <Animated.View style={[styles.card, animStyle(infoAnim)]}>
                <View style={styles.moistureHeader}>
                  <Text style={styles.cardTitle}>Soil Moisture</Text>
                  <Text style={styles.moistureTimestamp}>
                    {formatTime(moistureUpdated)}
                  </Text>
                </View>
                <View style={styles.cardDivider} />

                <View style={styles.moistureRow}>
                  <View style={styles.moistureValueBox}>
                    <Text
                      style={[
                        styles.moistureValue,
                        { color: getMoistureColor(moisture) },
                      ]}
                    >
                      {moisture !== null ? `${moisture}%` : "--"}
                    </Text>
                    <Text
                      style={[
                        styles.moistureLabel,
                        { color: getMoistureColor(moisture) },
                      ]}
                    >
                      {getMoistureLabel(moisture)}
                    </Text>
                  </View>

                  <View style={styles.moistureBarContainer}>
                    <View style={styles.moistureBarBg}>
                      <View
                        style={[
                          styles.moistureBarFill,
                          {
                            width: `${moisture || 0}%`,
                            backgroundColor: getMoistureColor(moisture),
                          },
                        ]}
                      />
                    </View>
                    <View style={styles.moistureBarLabels}>
                      <Text style={styles.moistureBarLabel}>Dry</Text>
                      <Text style={styles.moistureBarLabel}>Wet</Text>
                    </View>
                  </View>
                </View>

                {getMoistureAdvice(moisture) && (
                  <View
                    style={[
                      styles.adviceBox,
                      { borderLeftColor: getMoistureColor(moisture) },
                    ]}
                  >
                    <Text
                      style={[
                        styles.adviceText,
                        { color: getMoistureColor(moisture) },
                      ]}
                    >
                      {getMoistureAdvice(moisture)}
                    </Text>
                  </View>
                )}
              </Animated.View>
            </>
          )}
        </ScrollView>

        {/* Unified modal — handles errors and confirmations */}
        <Modal
          visible={modal.visible}
          animationType="fade"
          transparent
          onRequestClose={closeModal}
        >
          <View style={styles.modalOverlay}>
            <View style={styles.modalCard}>
              <View style={styles.modalHeader}>
                <Ionicons
                  name={modal.type === "confirm" ? "help-circle" : "alert-circle"}
                  size={22}
                  color={modal.type === "confirm" ? "#2e7d32" : "#e53935"}
                />
                <Text style={styles.modalTitle}>{modal.title}</Text>
              </View>
              <Text style={styles.modalMessage}>{modal.message}</Text>

              {modal.type === "confirm" ? (
                <View style={styles.modalBtnRow}>
                  <TouchableOpacity
                    style={styles.modalBtnOutline}
                    onPress={closeModal}
                  >
                    <Text style={styles.modalBtnOutlineText}>Cancel</Text>
                  </TouchableOpacity>
                  <TouchableOpacity
                    style={styles.modalBtnSolid}
                    onPress={modal.onConfirm}
                  >
                    <Text style={styles.modalBtnSolidText}>Confirm</Text>
                  </TouchableOpacity>
                </View>
              ) : (
                <TouchableOpacity
                  style={styles.modalBtnDanger}
                  onPress={closeModal}
                >
                  <Text style={styles.modalBtnSolidText}>Dismiss</Text>
                </TouchableOpacity>
              )}
            </View>
          </View>
        </Modal>
      </View>
    );
  }

  const styles = StyleSheet.create({
    screen: { flex: 1, backgroundColor: "#f5f5f5" },
    scrollView: { flex: 1 },
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
    content: { padding: 16, paddingBottom: 40 },
    statusBar: {
      flexDirection: "row",
      alignItems: "center",
      padding: 12,
      borderRadius: 10,
      marginBottom: 16,
      gap: 8,
    },
    statusDot: { width: 8, height: 8, borderRadius: 4 },
    statusText: { flex: 1, fontSize: 13, fontWeight: "500" },
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
    pumpIndicator: {
      width: 140,
      height: 140,
      borderRadius: 70,
      justifyContent: "center",
      alignItems: "center",
      alignSelf: "center",
      marginBottom: 20,
    },
    pumpDot: { width: 20, height: 20, borderRadius: 10, marginBottom: 8 },
    pumpStatus: { fontSize: 18, fontWeight: "bold", letterSpacing: 2 },
    toggleButton: {
      width: "100%",
      paddingVertical: 14,
      borderRadius: 12,
      alignItems: "center",
      marginBottom: 10,
    },
    buttonDisabled: { opacity: 0.6 },
    toggleText: { color: "#fff", fontSize: 15, fontWeight: "700" },
    lastUpdated: { fontSize: 12, color: "#aaa", textAlign: "center" },
    autoRow: { flexDirection: "row", alignItems: "center", gap: 12 },
    autoTextGroup: { flex: 1 },
    autoSubtitle: { fontSize: 12, color: "#999", marginTop: 4 },
    moistureHeader: {
      flexDirection: "row",
      alignItems: "center",
      justifyContent: "space-between",
    },
    moistureTimestamp: { fontSize: 11, color: "#bbb" },
    moistureRow: { flexDirection: "row", alignItems: "center", gap: 16 },
    moistureValueBox: { alignItems: "center", width: 80 },
    moistureValue: { fontSize: 32, fontWeight: "800" },
    moistureLabel: { fontSize: 13, fontWeight: "600", marginTop: 2 },
    moistureBarContainer: { flex: 1 },
    moistureBarBg: {
      height: 12,
      borderRadius: 6,
      backgroundColor: "#f0f0f0",
      overflow: "hidden",
    },
    moistureBarFill: { height: "100%", borderRadius: 6 },
    moistureBarLabels: {
      flexDirection: "row",
      justifyContent: "space-between",
      marginTop: 4,
    },
    moistureBarLabel: { fontSize: 10, color: "#bbb" },
    adviceBox: {
      marginTop: 14,
      padding: 12,
      borderRadius: 10,
      borderLeftWidth: 3,
      backgroundColor: "#fafafa",
    },
    adviceText: { fontSize: 13, fontWeight: "600" },
    modalOverlay: {
      flex: 1,
      backgroundColor: "rgba(0,0,0,0.5)",
      justifyContent: "center",
      alignItems: "center",
      padding: 24,
    },
    modalCard: {
      backgroundColor: "#fff",
      borderRadius: 24,
      padding: 24,
      width: "100%",
      maxWidth: 320,
      elevation: 10,
      shadowColor: "#000",
      shadowOffset: { width: 0, height: 4 },
      shadowOpacity: 0.1,
      shadowRadius: 12,
    },
    modalHeader: {
      flexDirection: "row",
      alignItems: "center",
      marginBottom: 15,
      borderBottomWidth: 1,
      borderBottomColor: "#f5f5f5",
      paddingBottom: 10,
      gap: 10,
    },
    modalTitle: { fontSize: 18, fontWeight: "800", color: "#333" },
    modalMessage: {
      fontSize: 14,
      color: "#666",
      lineHeight: 22,
      marginBottom: 20,
    },
    modalBtnRow: { flexDirection: "row", gap: 10 },
    modalBtnOutline: {
      flex: 1,
      height: 48,
      borderRadius: 12,
      borderWidth: 1.5,
      borderColor: "#e0e0e0",
      alignItems: "center",
      justifyContent: "center",
    },
    modalBtnOutlineText: { color: "#666", fontWeight: "700", fontSize: 15 },
    modalBtnSolid: {
      flex: 1,
      height: 48,
      borderRadius: 12,
      backgroundColor: "#2e7d32",
      alignItems: "center",
      justifyContent: "center",
    },
    modalBtnDanger: {
      height: 48,
      borderRadius: 12,
      backgroundColor: "#e53935",
      alignItems: "center",
      justifyContent: "center",
    },
    modalBtnSolidText: { color: "#fff", fontWeight: "700", fontSize: 15 },
  });
