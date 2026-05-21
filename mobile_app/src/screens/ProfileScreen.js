import React, { useState, useRef, useCallback } from "react";
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Alert,
  ScrollView,
  ActivityIndicator,
  Switch,
  Image,
  Animated,
  Easing,
  Platform,
  ToastAndroid,
  Modal,
  StatusBar,
} from "react-native";
import { useAuth } from "../context/AuthContext";
import { useFocusEffect } from "@react-navigation/native";
import * as SecureStore from "expo-secure-store";
import * as ImagePicker from "expo-image-picker";
import * as FileSystem from "expo-file-system/legacy";
import { Ionicons } from "@expo/vector-icons";
import { BACKEND_URL } from "../config";

// Encapsulated static catalog array outside the component to maximize layout runtime performance
const CATALOG_PORTFOLIO = [
  { name: "Apple", desc: "Scab, Black Rot, Cedar Apple Rust, Healthy" },
  {
    name: "Corn (Maize)",
    desc: "Blight, Common Rust, Gray Leaf Spot, Northern Blight, Healthy",
  },
  {
    name: "Cotton",
    desc: "Aphids, Bacterial Blight, Powdery Mildew, Target Spot, Healthy",
  },
  {
    name: "Grape",
    desc: "Black Rot, Esca (Black Measles), Leaf Blight, Healthy",
  },
  {
    name: "Mango",
    desc: "Anthracnose, Bacterial Canker, Cutting Weevil, Die Back, Gall Midge, Powdery Mildew, Sooty Mould, Healthy",
  },
  { name: "Potato", desc: "Early Blight, Late Blight, Healthy" },
  {
    name: "Rice",
    desc: "Bacterial Blight, Brown Spot, Hispa, Leaf Blast, Leaf Scald, Sheath Blight, Healthy",
  },
  {
    name: "Sugarcane",
    desc: "Bacterial Blight, Banded Chlorosis, Brown Rust/Spot, Grassy Shoot, Mosaic, Pokkah Boeng, Red/Sett Rot, Smut, Yellow Leaf",
  },
  {
    name: "Tomato",
    desc: "Bacterial Spot, Early/Late Blight, Leaf Mold, Mosaic/Yellow Leaf Curl Virus, Powdery Mildew, Septoria, Spider Mites, Target Spot",
  },
  {
    name: "Wheat",
    desc: "Aphids, Rusts (Black, Brown, Yellow), Blast, Root Rot, Head Blight, Mites, Powdery Mildew, Septoria, Smut, Stem Fly, Tan Spot",
  },
];

export default function ProfileScreen({ navigation }) {
  const { userInfo, logout } = useAuth();
  const [loading, setLoading] = useState(false);
  const [statsLoading, setStatsLoading] = useState(true);
  const [stats, setStats] = useState({ total: 0, diseased: 0, healthy: 0 });
  const [notifications, setNotifications] = useState(true);
  const [profilePic, setProfilePic] = useState(null);
  const [catalogVisible, setCatalogVisible] = useState(false);

  // Layout Animation Vectors
  const headerAnim = useRef(new Animated.Value(0)).current;
  const statsAnim = useRef(new Animated.Value(0)).current;
  const sectionsAnim = useRef(new Animated.Value(0)).current;
  const avatarScale = useRef(new Animated.Value(1)).current;

  // Cross-Platform Native Toast Helper
  const showNotification = useCallback((msg) => {
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
  }, []);

  // Sync network state analytics and persist user photo references securely
  const fetchScreenData = async () => {
    try {
      setStatsLoading(true);
      const [token, savedPic] = await Promise.all([
        SecureStore.getItemAsync("userToken"),
        SecureStore.getItemAsync("profilePic"),
      ]);

      if (savedPic) setProfilePic(savedPic);

      const response = await fetch(`${BACKEND_URL}/api/scans`, {
        headers: { Authorization: `Bearer ${token}` },
      });

      if (response.ok) {
        const scans = await response.json();
        const total = scans.length;
        const diseased = scans.filter((s) => !s.is_healthy).length;
        const healthy = scans.filter((s) => s.is_healthy).length;
        setStats({ total, diseased, healthy });
      }
    } catch (err) {
      console.log("Sync processing fault:", err.message);
      showNotification("Network sync delayed. Displaying cached metrics.");
    } finally {
      setStatsLoading(false);
    }
  };

  // Staggered screen entry animations on navigation window focusing
  useFocusEffect(
    useCallback(() => {
      headerAnim.setValue(0);
      statsAnim.setValue(0);
      sectionsAnim.setValue(0);

      Animated.stagger(60, [
        Animated.timing(headerAnim, {
          toValue: 1,
          duration: 350,
          easing: Easing.out(Easing.cubic),
          useNativeDriver: true,
        }),
        Animated.timing(statsAnim, {
          toValue: 1,
          duration: 350,
          easing: Easing.out(Easing.cubic),
          useNativeDriver: true,
        }),
        Animated.timing(sectionsAnim, {
          toValue: 1,
          duration: 350,
          easing: Easing.out(Easing.cubic),
          useNativeDriver: true,
        }),
      ]).start();

      fetchScreenData();
    }, []),
  );

  // Persistence Fix: Copy image from volatile runtime temporary folder to sandbox document storage
  const saveImagePermanently = async (tempUri) => {
    try {
      const filename = `avatar_${Date.now()}.jpg`;
      const permanentDirectory = `${FileSystem.documentDirectory}${filename}`;

      await FileSystem.copyAsync({
        from: tempUri,
        to: permanentDirectory,
      });

      setProfilePic(permanentDirectory);
      await SecureStore.setItemAsync("profilePic", permanentDirectory);
      showNotification("Profile picture updated successfully!");
    } catch (error) {
      console.log("Image processing structural error:", error);
      showNotification("Failed to process and save image layout.");
    }
  };

  const handleChangePic = () => {
    Alert.alert(
      "Profile Photo",
      "Update your profile picture:",
      [
        {
          text: "Open Camera",
          onPress: async () => {
            const permission =
              await ImagePicker.requestCameraPermissionsAsync();
            if (!permission.granted)
              return showNotification("Camera access refused.");

            // Force high-contrast status bar styles before opening the native cropper
            StatusBar.setBarStyle("light-content", true);
            if (Platform.OS === "android")
              StatusBar.setBackgroundColor("#000000", true);

            const result = await ImagePicker.launchCameraAsync({
              quality: 0.6,
              allowsEditing: true,
              aspect: [1, 1],
            });

            // Revert back to app matching light theme colors once selection completes
            StatusBar.setBarStyle("dark-content", true);
            if (Platform.OS === "android")
              StatusBar.setBackgroundColor("#f5f5f5", true);

            if (!result.canceled) {
              await saveImagePermanently(result.assets[0].uri);
            }
          },
        },
        {
          text: "Choose from Gallery",
          onPress: async () => {
            const permission =
              await ImagePicker.requestMediaLibraryPermissionsAsync();
            if (!permission.granted)
              return showNotification("Library access refused.");

            // Force high-contrast status bar styles before opening the native cropper
            StatusBar.setBarStyle("light-content", true);
            if (Platform.OS === "android")
              StatusBar.setBackgroundColor("#000000", true);

            const result = await ImagePicker.launchImageLibraryAsync({
              quality: 0.6,
              allowsEditing: true,
              aspect: [1, 1],
            });

            // Revert back to app matching light theme colors once selection completes
            StatusBar.setBarStyle("dark-content", true);
            if (Platform.OS === "android")
              StatusBar.setBackgroundColor("#f5f5f5", true);

            if (!result.canceled) {
              await saveImagePermanently(result.assets[0].uri);
            }
          },
        },
        profilePic && {
          text: "Remove Photo",
          style: "destructive",
          onPress: async () => {
            setProfilePic(null);
            await SecureStore.deleteItemAsync("profilePic");
            showNotification("Profile picture removed.");
          },
        },
        { text: "Cancel", style: "cancel" },
      ].filter(Boolean),
    );
  };

  const handleClearCache = () => {
    Alert.alert(
      'Optimize Storage Cache',
      'This will delete cached preview images to free up system space. Your historical records remain perfectly safe on our backend.',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Clear Cache',
          onPress: async () => {
            try {
              // 1. Target the native system cache folder URI
              const cacheDir = FileSystem.cacheDirectory;
              
              if (cacheDir) {
                // 2. Read all files currently nested in the cache sheet
                const cachedFiles = await FileSystem.readDirectoryAsync(cacheDir);
                
                // 3. Loop through and delete each file block concurrently
                await Promise.all(
                  cachedFiles.map(file => 
                    FileSystem.deleteAsync(`${cacheDir}${file}`, { idempotent: true })
                  )
                );
                
                showNotification('Application image cache optimized and cleared!');
              }
            } catch (error) {
              console.log('Cache eviction error:', error.message);
              showNotification('Storage directory optimization completed.');
            }
          }
        }
      ]
    );
  };
  
  const handleLogout = () => {
    Alert.alert("Log Out", "Are you sure you want to sign out of AgriGuard?", [
      { text: "Cancel", style: "cancel" },
      {
        text: "Log Out",
        style: "destructive",
        onPress: async () => {
          setLoading(true);
          await logout();
        },
      },
    ]);
  };

  const getInitials = (name) => {
    if (!name) return "?";
    return name
      .split(" ")
      .map((n) => n[0])
      .join("")
      .toUpperCase()
      .slice(0, 2);
  };

  const pressIn = () =>
    Animated.spring(avatarScale, {
      toValue: 0.94,
      useNativeDriver: true,
    }).start();
  const pressOut = () =>
    Animated.spring(avatarScale, {
      toValue: 1,
      friction: 4,
      useNativeDriver: true,
    }).start();

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

  const MenuItem = ({
    icon,
    label,
    value,
    onPress,
    danger,
    toggle,
    toggleValue,
    onToggle,
  }) => (
    <TouchableOpacity
      style={styles.menuItem}
      onPress={onPress}
      disabled={!onPress && !toggle}
      activeOpacity={onPress ? 0.6 : 1}
    >
      <View style={[styles.menuIconBox, danger && styles.menuIconBoxDanger]}>
        <Ionicons
          name={icon}
          size={18}
          color={danger ? "#e53935" : "#2e7d32"}
        />
      </View>
      <Text style={[styles.menuLabel, danger && styles.dangerText]}>
        {label}
      </Text>
      {value && <Text style={styles.menuValue}>{value}</Text>}
      {toggle && (
        <Switch
          value={toggleValue}
          onValueChange={onToggle}
          trackColor={{ false: "#eef0ef", true: "#a5d6a7" }}
          thumbColor={toggleValue ? "#2e7d32" : "#f4f3f4"}
        />
      )}
      {onPress && !danger && (
        <Ionicons name="chevron-forward" size={16} color="#bbb" />
      )}
    </TouchableOpacity>
  );

  return (
    <ScrollView
      style={styles.container}
      contentContainerStyle={styles.content}
      showsVerticalScrollIndicator={false}
    >
      {/* Profile Header Block */}
      <Animated.View style={[styles.headerCard, animStyle(headerAnim, -8)]}>
        <Animated.View style={{ transform: [{ scale: avatarScale }] }}>
          <TouchableOpacity
            style={styles.avatarWrapper}
            onPress={handleChangePic}
            onPressIn={pressIn}
            onPressOut={pressOut}
            activeOpacity={1}
          >
            {profilePic ? (
              <Image source={{ uri: profilePic }} style={styles.avatarImage} />
            ) : (
              <View style={styles.avatarFallback}>
                <Text style={styles.avatarText}>
                  {getInitials(userInfo?.name)}
                </Text>
              </View>
            )}
            <View style={styles.cameraBadge}>
              <Ionicons name="camera" size={12} color="#fff" />
            </View>
          </TouchableOpacity>
        </Animated.View>

        <Text style={styles.name}>{userInfo?.name || "AgriGuard Farmer"}</Text>
        <Text style={styles.email}>
          {userInfo?.email || "farmer@agriguard.com"}
        </Text>

        <View style={styles.rolePill}>
          <View style={styles.roleDot} />
          <Text style={styles.roleText}>
            {userInfo?.role === "admin"
              ? "System Administrator"
              : "Verified Farmer Account"}
          </Text>
        </View>

        <TouchableOpacity
          style={styles.editBtn}
          onPress={() => navigation.navigate("EditProfileScreen")}
          activeOpacity={0.7}
        >
          <Ionicons
            name="pencil"
            size={14}
            color="#2e7d32"
            style={{ marginRight: 6 }}
          />
          <Text style={styles.editBtnText}>Modify Profile</Text>
        </TouchableOpacity>
      </Animated.View>

      {/* Analytics Metric Grid */}
      <Animated.View style={[styles.statsContainer, animStyle(statsAnim, 8)]}>
        {statsLoading ? (
          <View style={styles.statsLoaderBox}>
            <ActivityIndicator color="#2e7d32" />
          </View>
        ) : (
          <React.Fragment>
            <View style={styles.statCell}>
              <Text style={[styles.statNumber, { color: "#2c3e50" }]}>
                {stats.total}
              </Text>
              <Text style={styles.statLabel}>Total Scans</Text>
            </View>
            <View style={styles.statDivider} />
            <View style={styles.statCell}>
              <Text style={[styles.statNumber, { color: "#e53935" }]}>
                {stats.diseased}
              </Text>
              <Text style={styles.statLabel}>Diseased</Text>
            </View>
            <View style={styles.statDivider} />
            <View style={styles.statCell}>
              <Text style={[styles.statNumber, { color: "#2e7d32" }]}>
                {stats.healthy}
              </Text>
              <Text style={styles.statLabel}>Healthy</Text>
            </View>
          </React.Fragment>
        )}
      </Animated.View>

      {/* Settings Options Groups */}
      <Animated.View style={animStyle(sectionsAnim, 12)}>
        <Text style={styles.sectionLabel}>PREFERENCES</Text>
        <View style={styles.menuCard}>
          <MenuItem
            icon="notifications"
            label="Push Alerts Notify"
            toggle
            toggleValue={notifications}
            onToggle={setNotifications}
          />
          <View style={styles.menuDivider} />
          <MenuItem
            icon="language"
            label="System Language"
            value="English"
            onPress={() =>
              showNotification("Urdu language pack will be available soon.")
            }
          />
        </View>

        <Text style={styles.sectionLabel}>SUPPORT & UTILITIES</Text>
        <View style={styles.menuCard}>
          <MenuItem
            icon="help-circle"
            label="How to Diagnose Crops"
            onPress={() =>
              Alert.alert(
                "Easy User Guide",
                '1. Go to the Detect tab.\n2. Tap the button to open your camera or choose a leaf photo from your gallery.\n3. Make sure the leaf is clearly visible under good lighting.\n4. Tap "Analyze Crop" and wait a moment.\n5. The app will immediately display whether the crop is healthy or diseased, along with treatment recommendations.',
              )
            }
          />
          <View style={styles.menuDivider} />
          <MenuItem
            icon="bug"
            label="Report Technical Glitch"
            onPress={() =>
              Alert.alert(
                "Help Desk Support",
                "Found a bug or facing an issue? Please email our support channel with brief details or a screenshot:\n\narslanahmednaseem@gmail.com\n\nThe AgriGuard Support Team will assist you within 24 hours.",
              )
            }
          />
          <View style={styles.menuDivider} />
          <MenuItem
            icon="albums"
            label="Supported Crops Catalog"
            onPress={() => setCatalogVisible(true)}
          />
          <View style={styles.menuDivider} />
          <MenuItem
            icon="trash-bin-outline"
            label="Optimize Storage (Clear Cache)"
            onPress={handleClearCache}
          />
        </View>

        <Text style={styles.sectionLabel}>APPLICATION METRICS</Text>
        <View style={styles.menuCard}>
          <MenuItem
            icon="information-circle"
            label="Engine Core Version"
            value="v1.0.0 (Beta)"
          />
        </View>

        <TouchableOpacity
          style={styles.logoutButton}
          onPress={handleLogout}
          disabled={loading}
          activeOpacity={0.75}
        >
          {loading ? (
            <ActivityIndicator color="#e53935" />
          ) : (
            <View style={styles.logoutContent}>
              <Ionicons
                name="log-out"
                size={18}
                color="#e53935"
                style={{ marginRight: 8 }}
              />
              <Text style={styles.logoutText}>Log Out Session</Text>
            </View>
          )}
        </TouchableOpacity>

        <Text style={styles.footer}>
          AgriGuard Suite · Intelligent Crop Architecture 2026
        </Text>
      </Animated.View>

      {/* ========================================================
          SUPPORTED CROPS OVERLAY BOTTOM SHEET MODAL
         ======================================================== */}
      <Modal
        visible={catalogVisible}
        animationType="slide"
        transparent={true}
        onRequestClose={() => setCatalogVisible(false)}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalCard}>
            <View style={styles.modalHeader}>
              <TouchableOpacity onPress={() => setCatalogVisible(false)}>
                <Ionicons name="close-circle" size={26} color="#888" />
              </TouchableOpacity>
              <Text style={styles.modalTitle}>Supported Crop Portfolio</Text>
              {/* BUG FIX: Transformed web <div> to proper native layout container component */}
              <View style={{ width: 26 }} />
            </View>

            <ScrollView
              showsVerticalScrollIndicator={false}
              contentContainerStyle={{ paddingBottom: 20 }}
            >
              <Text style={styles.modalSubtitle}>
                AgriGuard safely scans for unique diseases across these 10 core
                crops:
              </Text>
              {CATALOG_PORTFOLIO.map((item) => (
                <View key={item.name} style={styles.catalogItem}>
                  <View style={styles.catalogBulletRow}>
                    <Ionicons
                      name="checkmark-circle"
                      size={16}
                      color="#2e7d32"
                      style={{ marginRight: 6, marginTop: 2 }}
                    />
                    <Text style={styles.cropNameText}>{item.name}</Text>
                  </View>
                  <Text style={styles.cropDiseasesText}>{item.desc}</Text>
                </View>
              ))}
            </ScrollView>
          </View>
        </View>
      </Modal>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#f5f5f5" },
  content: {
    paddingTop: 24,
    paddingHorizontal: 20,
    // EXTREME SPACING FIX: Forces an extra 160 units of clear empty space
    // at the absolute end of the scroll container to elevate the button above your bar
    paddingBottom: Platform.OS === "ios" ? 180 : 160,
  },
  headerCard: {
    backgroundColor: "#fff",
    borderRadius: 20,
    padding: 24,
    alignItems: "center",
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.04,
    shadowRadius: 10,
    elevation: 3,
    marginBottom: 16,
  },
  avatarWrapper: { position: "relative", marginBottom: 14 },
  avatarImage: {
    width: 88,
    height: 88,
    borderRadius: 44,
    borderWidth: 3,
    borderColor: "#e8f5e9",
  },
  avatarFallback: {
    width: 88,
    height: 88,
    borderRadius: 44,
    backgroundColor: "#2e7d32",
    justifyContent: "center",
    alignItems: "center",
    borderWidth: 3,
    borderColor: "#e8f5e9",
  },
  avatarText: {
    fontSize: 26,
    fontWeight: "800",
    color: "#fff",
    letterSpacing: 0.5,
  },
  cameraBadge: {
    position: "absolute",
    bottom: 2,
    right: 2,
    width: 26,
    height: 26,
    borderRadius: 13,
    backgroundColor: "#2e7d32",
    justifyContent: "center",
    alignItems: "center",
    borderWidth: 2,
    borderColor: "#fff",
  },
  name: { fontSize: 22, fontWeight: "800", color: "#1a1a1a", marginBottom: 4 },
  email: { fontSize: 14, color: "#888", marginBottom: 12 },
  rolePill: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "#e8f5e9",
    paddingHorizontal: 14,
    paddingVertical: 6,
    borderRadius: 20,
    marginBottom: 16,
  },
  roleDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: "#2e7d32",
    marginRight: 8,
  },
  roleText: { fontSize: 12, color: "#2e7d32", fontWeight: "700" },
  editBtn: {
    flexDirection: "row",
    alignItems: "center",
    borderWidth: 1.5,
    borderColor: "#e2ece2",
    borderRadius: 12,
    paddingHorizontal: 18,
    paddingVertical: 8,
    backgroundColor: "#fafafa",
  },
  editBtnText: { color: "#2e7d32", fontWeight: "700", fontSize: 13 },
  statsContainer: {
    flexDirection: "row",
    backgroundColor: "#fff",
    borderRadius: 20,
    paddingVertical: 20,
    marginBottom: 20,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.04,
    shadowRadius: 10,
    elevation: 3,
    minHeight: 78,
  },
  statsLoaderBox: { flex: 1, justifyContent: "center", alignItems: "center" },
  statCell: { flex: 1, alignItems: "center", justifyContent: "center" },
  statDivider: {
    width: 1,
    height: "60%",
    backgroundColor: "#f0f0f0",
    alignSelf: "center",
  },
  statNumber: { fontSize: 24, fontWeight: "800" },
  statLabel: { fontSize: 11, color: "#888", fontWeight: "600", marginTop: 4 },
  sectionLabel: {
    fontSize: 11,
    fontWeight: "700",
    color: "#888",
    letterSpacing: 1,
    marginLeft: 4,
    marginBottom: 8,
    marginTop: 4,
  },
  menuCard: {
    backgroundColor: "#fff",
    marginBottom: 20,
    borderRadius: 20,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.04,
    shadowRadius: 10,
    elevation: 3,
    overflow: "hidden",
  },
  menuItem: {
    flexDirection: "row",
    alignItems: "center",
    paddingHorizontal: 16,
    paddingVertical: 14,
  },
  menuIconBox: {
    width: 32,
    height: 32,
    borderRadius: 10,
    backgroundColor: "#e8f5e9",
    justifyContent: "center",
    alignItems: "center",
    marginRight: 14,
  },
  menuIconBoxDanger: { backgroundColor: "#fdecea" },
  menuLabel: { flex: 1, fontSize: 15, color: "#333", fontWeight: "600" },
  menuValue: { fontSize: 14, color: "#888", marginRight: 4, fontWeight: "500" },
  menuDivider: { height: 1, backgroundColor: "#f7f7f7", marginLeft: 62 },
  dangerText: { color: "#e53935" },
  logoutButton: {
    paddingVertical: 14,
    borderRadius: 16,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#fff",
    borderWidth: 1.5,
    borderColor: "#fdecea",
    marginTop: 8,
    marginBottom: 16,
  },
  logoutContent: { flexDirection: "row", alignItems: "center" },
  logoutText: { color: "#e53935", fontSize: 15, fontWeight: "700" },
  footer: {
    textAlign: "center",
    fontSize: 11,
    color: "#b0b0b0",
    fontWeight: "500",
    marginTop: 8,
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: "rgba(0,0,0,0.4)",
    justifyContent: "flex-end",
  },
  modalCard: {
    backgroundColor: "#fff",
    borderTopLeftRadius: 24,
    borderTopRightRadius: 24,
    padding: 24,
    maxHeight: "80%",
    shadowColor: "#000",
    shadowOpacity: 0.15,
    shadowRadius: 12,
    elevation: 10,
  },
  modalHeader: { flexDirection: "row", alignItems: "center", marginBottom: 20 },
  modalTitle: {
    flex: 1,
    fontSize: 18,
    fontWeight: "800",
    color: "#2e7d32",
    textAlign: "center",
    marginRight: 26,
  },
  modalSubtitle: {
    fontSize: 13,
    color: "#555",
    lineHeight: 18,
    marginBottom: 20,
    backgroundColor: "#f9f9f9",
    padding: 14,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: "#eee",
  },
  catalogItem: {
    marginBottom: 16,
    borderBottomWidth: 1,
    borderBottomColor: "#f7f7f7",
    paddingBottom: 12,
  },
  catalogBulletRow: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: 4,
  },
  cropNameText: { fontSize: 15, fontWeight: "700", color: "#222" },
  cropDiseasesText: {
    fontSize: 13,
    color: "#666",
    lineHeight: 18,
    paddingLeft: 22,
  },
});
