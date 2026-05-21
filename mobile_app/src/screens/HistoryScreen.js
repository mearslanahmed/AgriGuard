import React, { useState, useCallback, useRef } from "react";
import {
  View,
  Text,
  StyleSheet,
  FlatList,
  TouchableOpacity,
  ActivityIndicator,
  RefreshControl,
  Animated,
  Easing,
  StatusBar,
  Platform,
  Modal,
  Alert,
} from "react-native";
import { useFocusEffect } from "@react-navigation/native";
import * as SecureStore from "expo-secure-store";
import { Ionicons } from "@expo/vector-icons";
import { BACKEND_URL } from "../config";

const PremiumCard = ({ item, index, onDelete }) => {
  const cardAnim = useRef(new Animated.Value(0)).current;

  React.useEffect(() => {
    Animated.timing(cardAnim, {
      toValue: 1,
      duration: 350,
      delay: index * 40,
      easing: Easing.out(Easing.cubic),
      useNativeDriver: true,
    }).start();
  }, []);

  const getConfidenceColor = (c) => {
    if (c >= 85) return "#2e7d32";
    if (c >= 60) return "#f57c00";
    return "#e53935";
  };

  return (
    <Animated.View
      style={[
        styles.card,
        {
          opacity: cardAnim,
          transform: [
            {
              translateY: cardAnim.interpolate({
                inputRange: [0, 1],
                outputRange: [20, 0],
              }),
            },
          ],
        },
      ]}
    >
      <View
        style={[
          styles.cardAccent,
          { backgroundColor: item.is_healthy ? "#2e7d32" : "#e53935" },
        ]}
      />

      <View style={styles.cardMain}>
        <View style={styles.cardHeaderRow}>
          <Text style={styles.cropText} numberOfLines={1}>
            {item.crop}
          </Text>
          <View style={styles.headerRightBlock}>
            <Text style={styles.dateText}>
              {new Date(item.createdAt).toLocaleDateString("en-PK", {
                day: "numeric",
                month: "short",
              })}
            </Text>
            <TouchableOpacity
              style={styles.cardDeleteBtn}
              onPress={() => onDelete(item._id)}
            >
              <Ionicons name="trash-outline" size={14} color="#e53935" />
            </TouchableOpacity>
          </View>
        </View>

        <Text style={styles.conditionText} numberOfLines={1}>
          {item.is_healthy ? "Healthy Condition" : item.disease}
        </Text>

        <View style={styles.cardFooterRow}>
          <Text style={styles.metricsLabel}>AI Metrics Confidence</Text>
          <Text
            style={[
              styles.metricsValue,
              { color: getConfidenceColor(item.confidence) },
            ]}
          >
            {item.confidence.toFixed(1)}%
          </Text>
        </View>
      </View>
    </Animated.View>
  );
};

export default function HistoryScreen({ onOpen }) {
  const [scans, setScans] = useState([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  // Custom Modal Overlay States
  const [deleteModalVisible, setDeleteModalVisible] = useState(false);
  const [selectedScanId, setSelectedScanId] = useState(null);
  const [deletingLoading, setDeletingLoading] = useState(false);

  // Screen Entrance Animation Vectors
  const headerAnim = useRef(new Animated.Value(0)).current;
  const listAnim = useRef(new Animated.Value(0)).current;

  useFocusEffect(
    useCallback(() => {
      if (onOpen) onOpen();
      fetchScans();

      headerAnim.setValue(0);
      listAnim.setValue(0);

      Animated.stagger(100, [
        Animated.timing(headerAnim, {
          toValue: 1,
          duration: 350,
          easing: Easing.out(Easing.cubic),
          useNativeDriver: true,
        }),
        Animated.timing(listAnim, {
          toValue: 1,
          duration: 400,
          easing: Easing.out(Easing.cubic),
          useNativeDriver: true,
        }),
      ]).start();
    }, [onOpen]),
  );

  const fetchScans = async () => {
    try {
      const token = await SecureStore.getItemAsync("userToken");
      const response = await fetch(`${BACKEND_URL}/api/scans`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      const data = await response.json();
      if (response.ok) setScans(data);
    } catch (err) {
      console.log("History data sync delayed:", err.message);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  const handleDeleteTrigger = (id) => {
    setSelectedScanId(id);
    setDeleteModalVisible(true);
  };

  const executeDeleteScan = async () => {
    setDeletingLoading(true);
    try {
      const token = await SecureStore.getItemAsync("userToken");
      const response = await fetch(
        `${BACKEND_URL}/api/scans/${selectedScanId}`,
        {
          method: "DELETE",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${token}`,
          },
        },
      );

      const data = await response.json();
      if (!response.ok)
        throw new Error(
          data.message || `Server error code: ${response.status}`,
        );

      setScans((prev) => prev.filter((s) => s._id !== selectedScanId));
      setDeleteModalVisible(false);
      setSelectedScanId(null);
    } catch (err) {
      Alert.alert("Backend Error", err.message);
      setDeleteModalVisible(false);
    } finally {
      setDeletingLoading(false);
    }
  };

  const animStyle = (anim, slide = 12) => ({
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

  if (loading) {
    return (
      <View style={styles.centered}>
        <ActivityIndicator size="large" color="#2e7d32" />
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {/* Clean Unified Flat Header Layout */}
      <Animated.View style={[styles.header, animStyle(headerAnim, -10)]}>
        <View style={{ width: 40 }} />
        <View style={styles.titleAreaRow}>
          <Text style={styles.headerTitle}>Scan History</Text>
          {scans.length > 0 && (
            <View style={styles.countBadge}>
              <Text style={styles.countText}>{scans.length}</Text>
            </View>
          )}
        </View>
        <View style={{ width: 40 }} />
      </Animated.View>

      {/* Main Content Layout Container */}
      <Animated.View style={[{ flex: 1 }, animStyle(listAnim, 16)]}>
        <FlatList
          data={scans}
          keyExtractor={(item) => item._id}
          renderItem={({ item, index }) => (
            <PremiumCard
              item={item}
              index={index}
              onDelete={handleDeleteTrigger}
            />
          )}
          contentContainerStyle={[
            styles.list,
            scans.length === 0 && styles.centeredList,
          ]}
          showsVerticalScrollIndicator={false}
          refreshControl={
            <RefreshControl
              refreshing={refreshing}
              onRefresh={() => {
                setRefreshing(true);
                fetchScans();
              }}
              colors={["#2e7d32"]}
            />
          }
          ListEmptyComponent={
            <View style={styles.emptyState}>
              <View style={styles.emptyIconBox}>
                <Ionicons name="leaf-outline" size={26} color="#2e7d32" />
              </View>
              <Text style={styles.emptyTitle}>No scans tracked yet</Text>
            </View>
          }
        />
      </Animated.View>

      {/* ========================================================
          CUSTOM HIGH-FIDELITY MODAL: DELETE CONFIRMATION
         ======================================================== */}
      <Modal
        visible={deleteModalVisible}
        animationType="fade"
        transparent={true}
        onRequestClose={() => !deletingLoading && setDeleteModalVisible(false)}
      >
        <View style={styles.modalOverlayCenter}>
          <View style={styles.popupCard}>
            <View style={styles.popupHeader}>
              <Ionicons name="trash" size={20} color="#e53935" />
              <Text style={styles.popupTitle}>Erase Scan History</Text>
            </View>
            <Text style={styles.popupInstruction}>
              Are you sure you want to permanently delete this record?
            </Text>
            <View style={styles.modalActionRow}>
              <TouchableOpacity
                style={[styles.popupBtn, styles.btnCancelSolid, { flex: 1 }]}
                onPress={() => setDeleteModalVisible(false)}
                disabled={deletingLoading}
              >
                <Text style={styles.popupBtnTextCancel}>Cancel</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[styles.popupBtn, styles.btnDangerSolid, { flex: 1 }]}
                onPress={executeDeleteScan}
                disabled={deletingLoading}
              >
                {deletingLoading ? (
                  <ActivityIndicator color="#fff" />
                ) : (
                  <Text style={styles.popupBtnTextDanger}>Delete</Text>
                )}
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#f5f5f5" },
  centered: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    backgroundColor: "#f5f5f5",
  },
  centeredList: { flexGrow: 1, justifyContent: "center", alignItems: "center" },

  // Header Component Styling
  header: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingHorizontal: 16,
    paddingTop:
      Platform.OS === "ios"
        ? 52
        : StatusBar.currentHeight
          ? StatusBar.currentHeight + 12
          : 16,
    paddingBottom: 14,
    backgroundColor: "#fff",
    borderBottomWidth: 1,
    borderBottomColor: "#eef0ef",
  },
  titleAreaRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 8,
    flex: 1,
  },
  headerTitle: {
    fontSize: 16,
    fontWeight: "800",
    color: "#2e7d32",
    textAlign: "center",
  },
  countBadge: {
    backgroundColor: "#e8f5e9",
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 10,
  },
  countText: { fontSize: 11, color: "#2e7d32", fontWeight: "800" },
  list: { paddingHorizontal: 16, paddingTop: 14, paddingBottom: 160 },

  // Compact Premium Card Architecture
  card: {
    backgroundColor: "#fff",
    borderRadius: 16,
    marginBottom: 10,
    flexDirection: "row",
    overflow: "hidden",
    elevation: 2,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.03,
    shadowRadius: 6,
    borderWidth: 1,
    borderColor: "#eef0ef",
  },
  cardAccent: { width: 4 },
  cardMain: { flex: 1, padding: 14 },
  cardHeaderRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 4,
  },
  cropText: {
    fontSize: 15,
    fontWeight: "800",
    color: "#1a1a1a",
    flex: 1,
    marginRight: 8,
  },
  headerRightBlock: { flexDirection: "row", alignItems: "center", gap: 10 },
  dateText: { fontSize: 12, color: "#aaa", fontWeight: "500" },
  cardDeleteBtn: {
    width: 26,
    height: 26,
    borderRadius: 6,
    backgroundColor: "#fdecea",
    justifyContent: "center",
    alignItems: "center",
  },
  conditionText: {
    fontSize: 13,
    color: "#555",
    fontWeight: "600",
    marginBottom: 10,
  },
  cardFooterRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    borderTopWidth: 1,
    borderTopColor: "#fbfbfb",
    paddingTop: 8,
  },
  metricsLabel: {
    fontSize: 9,
    color: "#bbb",
    fontWeight: "800",
    textTransform: "uppercase",
    letterSpacing: 0.5,
  },
  metricsValue: { fontSize: 13, fontWeight: "800" },

  // Empty state configurations
  emptyState: { alignItems: "center", justifyContent: "center" },
  emptyIconBox: {
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: "#e8f5e9",
    justifyContent: "center",
    alignItems: "center",
    marginBottom: 10,
  },
  emptyTitle: { fontSize: 14, fontWeight: "700", color: "#777" },

  // Modals Styling Configurations
  modalOverlayCenter: {
    flex: 1,
    backgroundColor: "rgba(0,0,0,0.5)",
    justifyContent: "center",
    alignItems: "center",
    padding: 24,
  },
  popupCard: {
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
  popupHeader: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: 15,
    borderBottomWidth: 1,
    borderBottomColor: "#f5f5f5",
    paddingBottom: 10,
  },
  popupTitle: {
    fontSize: 18,
    fontWeight: "800",
    color: "#333",
    marginLeft: 10,
  },
  popupInstruction: {
    fontSize: 14,
    color: "#666",
    lineHeight: 22,
    marginBottom: 20,
  },
  popupBtn: {
    height: 48,
    borderRadius: 12,
    alignItems: "center",
    justifyContent: "center",
    flexDirection: "row",
  },
  btnDangerSolid: { backgroundColor: "#e53935" },
  btnCancelSolid: { backgroundColor: "#eee" },
  popupBtnTextDanger: { color: "#fff", fontWeight: "700", fontSize: 15 },
  popupBtnTextCancel: { color: "#666", fontWeight: "700", fontSize: 15 },
  modalActionRow: { flexDirection: "row", gap: 10, width: "100%" },
});
